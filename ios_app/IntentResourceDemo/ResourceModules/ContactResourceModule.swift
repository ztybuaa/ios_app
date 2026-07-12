import Contacts
import Foundation

final class ContactResourceModule {
    private let store = CNContactStore()
    private let cacheLock = NSLock()
    private var cachedContacts: [CNContact]?
    private var cacheGeneration: UInt64 = 0
    private var storeChangeObserver: NSObjectProtocol?

    init() {
        storeChangeObserver = NotificationCenter.default.addObserver(
            forName: .CNContactStoreDidChange,
            object: nil,
            queue: nil
        ) { [weak self] _ in
            self?.invalidateCache()
        }
    }

    deinit {
        if let storeChangeObserver {
            NotificationCenter.default.removeObserver(storeChangeObserver)
        }
    }

    func searchContacts(slots: NormalizedSlots, limit: Int = 10) async throws -> [ResourceCandidate] {
        if !slots.qualifiers.time.isEmpty || slots.qualifiers.selectionHint.contains("recent") {
            throw DemoError.resourceUnavailable(
                "iOS 通讯录不提供联系人创建时间，无法可靠执行“最近/今天新增”筛选。"
            )
        }

        guard let keyword = slots.searchKeyword?.trimmingCharacters(in: .whitespacesAndNewlines),
              !keyword.isEmpty else {
            throw DemoError.resourceUnavailable(
                "联系人请求缺少姓名、昵称、电话、邮箱或组织等检索条件；当前联系人选择上下文尚未接入。"
            )
        }
        return try await searchContacts(keyword: keyword, limit: limit)
    }

    func searchContacts(keyword: String, limit: Int = 10) async throws -> [ResourceCandidate] {
        try await ensureAccess()
        let subject = keyword
            .replacingOccurrences(of: "的", with: " ")
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !subject.isEmpty else { return [] }

        let contacts = try contactsSnapshot()
        return contacts
            .compactMap { makeCandidate(contact: $0, subject: subject) }
            .sorted { lhs, rhs in
                lhs.score == rhs.score ? lhs.title < rhs.title : lhs.score > rhs.score
            }
            .prefix(limit)
            .map { $0 }
    }

    private func ensureAccess() async throws {
        let status = CNContactStore.authorizationStatus(for: .contacts)
        switch status {
        case .authorized, .limited:
            return
        case .notDetermined:
            let granted = try await store.requestAccess(for: .contacts)
            if granted { return }
            throw DemoError.permissionDenied("通讯录权限被拒绝，无法检索联系人候选。")
        case .denied, .restricted:
            throw DemoError.permissionDenied("通讯录权限不可用，请在系统设置中允许访问联系人。")
        @unknown default:
            throw DemoError.permissionDenied("通讯录权限状态未知，无法继续检索。")
        }
    }

    private func contactsSnapshot() throws -> [CNContact] {
        cacheLock.lock()
        if let cachedContacts {
            cacheLock.unlock()
            return cachedContacts
        }
        let generation = cacheGeneration
        cacheLock.unlock()

        let request = CNContactFetchRequest(keysToFetch: Self.contactKeys)
        request.sortOrder = .userDefault
        var contacts: [CNContact] = []
        try store.enumerateContacts(with: request) { contact, _ in
            contacts.append(contact)
        }

        cacheLock.lock()
        guard generation == cacheGeneration else {
            cacheLock.unlock()
            return try contactsSnapshot()
        }
        self.cachedContacts = contacts
        cacheLock.unlock()
        return contacts
    }

    private func invalidateCache() {
        cacheLock.lock()
        cacheGeneration &+= 1
        cachedContacts = nil
        cacheLock.unlock()
    }

    private func makeCandidate(contact: CNContact, subject: String) -> ResourceCandidate? {
        let displayName = displayName(for: contact)
        let alternateName = [contact.givenName, contact.middleName, contact.familyName]
            .filter { !$0.isEmpty }
            .joined()
        let phones = contact.phoneNumbers.map { $0.value.stringValue }
        let emails = contact.emailAddresses.map { String($0.value) }
        let phoneticNames = [
            contact.phoneticFamilyName,
            contact.phoneticGivenName,
            contact.phoneticMiddleName,
            [contact.phoneticFamilyName, contact.phoneticGivenName].joined(separator: " ")
        ].filter { !$0.isEmpty } + latinVariants([displayName, alternateName, contact.nickname])
        let organizations = [contact.organizationName, contact.departmentName].filter { !$0.isEmpty }
        let roles = [contact.jobTitle].filter { !$0.isEmpty }
        let addresses = contact.postalAddresses.map {
            CNPostalAddressFormatter.string(from: $0.value, style: .mailingAddress)
        }
        let socialProfiles = contact.socialProfiles.flatMap { labeledValue in
            let profile = labeledValue.value
            return [profile.username, profile.userIdentifier, profile.urlString, profile.service]
                .filter { !$0.isEmpty }
        }
        let instantMessages = contact.instantMessageAddresses.flatMap { labeledValue in
            let address = labeledValue.value
            return [address.username, address.service].filter { !$0.isEmpty }
        }

        let fields = [
            CandidateTextField(name: "name", values: [displayName, alternateName].filter { !$0.isEmpty }, weight: 8.0),
            CandidateTextField(name: "nickname", values: [contact.nickname].filter { !$0.isEmpty }, weight: 7.0),
            CandidateTextField(name: "phonetic", values: phoneticNames, weight: 7.0),
            CandidateTextField(name: "phone", values: phones, weight: 10.0),
            CandidateTextField(name: "email", values: emails, weight: 10.0),
            CandidateTextField(name: "organization", values: organizations, weight: 4.5),
            CandidateTextField(name: "job", values: roles, weight: 5.0),
            CandidateTextField(name: "address", values: addresses, weight: 3.5),
            CandidateTextField(name: "social", values: socialProfiles + instantMessages, weight: 4.0)
        ]
        guard let match = CandidateScorer.lexicalScore(subject: subject, fields: fields) else {
            return nil
        }

        let details = [contact.organizationName, contact.departmentName, contact.jobTitle, phones.first, emails.first]
            .compactMap { $0 }
            .filter { !$0.isEmpty }
        return ResourceCandidate(
            id: contact.identifier,
            kind: .contact,
            title: displayName.isEmpty ? "未命名联系人" : displayName,
            subtitle: phones.first ?? emails.first ?? contact.organizationName,
            detail: details.joined(separator: " · "),
            score: match.value,
            debugInfo: "field-aware contact match fields=\(match.matchedFields.joined(separator: ","))"
        )
    }

    private func latinVariants(_ values: [String]) -> [String] {
        var variants: [String] = []
        for value in values where !value.isEmpty {
            guard let transformed = value.applyingTransform(.toLatin, reverse: false) else { continue }
            let folded = transformed
                .folding(options: [.diacriticInsensitive, .widthInsensitive], locale: Locale(identifier: "en_US_POSIX"))
                .lowercased()
            variants.append(folded)
            variants.append(folded.replacingOccurrences(of: " ", with: ""))
        }
        return Array(Set(variants.filter { !$0.isEmpty }))
    }

    private func displayName(for contact: CNContact) -> String {
        if let formatted = CNContactFormatter.string(from: contact, style: .fullName), !formatted.isEmpty {
            return formatted
        }
        if !contact.nickname.isEmpty { return contact.nickname }
        if !contact.organizationName.isEmpty { return contact.organizationName }
        if let phone = contact.phoneNumbers.first?.value.stringValue, !phone.isEmpty { return phone }
        if let email = contact.emailAddresses.first { return String(email.value) }
        return ""
    }

    private static let contactKeys: [CNKeyDescriptor] = [
        CNContactIdentifierKey as CNKeyDescriptor,
        CNContactFormatter.descriptorForRequiredKeys(for: .fullName),
        CNContactGivenNameKey as CNKeyDescriptor,
        CNContactMiddleNameKey as CNKeyDescriptor,
        CNContactFamilyNameKey as CNKeyDescriptor,
        CNContactPhoneticGivenNameKey as CNKeyDescriptor,
        CNContactPhoneticMiddleNameKey as CNKeyDescriptor,
        CNContactPhoneticFamilyNameKey as CNKeyDescriptor,
        CNContactNicknameKey as CNKeyDescriptor,
        CNContactOrganizationNameKey as CNKeyDescriptor,
        CNContactDepartmentNameKey as CNKeyDescriptor,
        CNContactJobTitleKey as CNKeyDescriptor,
        CNContactPhoneNumbersKey as CNKeyDescriptor,
        CNContactEmailAddressesKey as CNKeyDescriptor,
        CNContactPostalAddressesKey as CNKeyDescriptor,
        CNContactSocialProfilesKey as CNKeyDescriptor,
        CNContactInstantMessageAddressesKey as CNKeyDescriptor
    ]
}
