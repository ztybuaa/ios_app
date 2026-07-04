import Contacts
import Foundation

final class ContactResourceModule {
    private let store = CNContactStore()

    func searchContacts(keyword: String, limit: Int = 10) async throws -> [ResourceCandidate] {
        try await ensureAccess()

        let keys: [CNKeyDescriptor] = [
            CNContactGivenNameKey as CNKeyDescriptor,
            CNContactFamilyNameKey as CNKeyDescriptor,
            CNContactNicknameKey as CNKeyDescriptor,
            CNContactOrganizationNameKey as CNKeyDescriptor,
            CNContactPhoneNumbersKey as CNKeyDescriptor,
            CNContactEmailAddressesKey as CNKeyDescriptor
        ]
        let request = CNContactFetchRequest(keysToFetch: keys)
        request.sortOrder = .userDefault

        var candidates: [ResourceCandidate] = []
        try store.enumerateContacts(with: request) { contact, _ in
            let candidate = self.makeCandidate(contact: contact, keyword: keyword)
            if candidate.score > 0 {
                candidates.append(candidate)
            }
        }

        return candidates
            .sorted { $0.score == $1.score ? $0.title < $1.title : $0.score > $1.score }
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

    private func makeCandidate(contact: CNContact, keyword: String) -> ResourceCandidate {
        let displayName = CNContactFormatter.string(from: contact, style: .fullName)
            ?? [contact.familyName, contact.givenName].joined()
        let phones = contact.phoneNumbers.map { $0.value.stringValue }
        let emails = contact.emailAddresses.map { String($0.value) }
        let searchable = ([displayName, contact.nickname, contact.organizationName] + phones + emails)
            .joined(separator: " ")
        let score = CandidateScorer.score(
            keyword: keyword,
            phrase: keyword,
            targetText: searchable,
            tags: []
        )

        return ResourceCandidate(
            id: contact.identifier,
            kind: .contact,
            title: displayName.isEmpty ? "未命名联系人" : displayName,
            subtitle: phones.first ?? emails.first ?? contact.organizationName,
            detail: searchable,
            score: score,
            debugInfo: "matched contact fields: name/nickname/org/phone/email"
        )
    }
}
