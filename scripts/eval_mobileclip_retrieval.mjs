import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  AutoProcessor,
  AutoTokenizer,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  RawImage,
  dot,
  env,
} from '@huggingface/transformers';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const datasetRoot = path.join(root, 'processed', 'eval', 'semantic_image_retrieval');
const manifestPath = path.join(datasetRoot, 'manifest.json');
const imageRoot = path.join(datasetRoot, 'images');
const resultRoot = path.join(datasetRoot, 'results');
const modelId = 'Xenova/mobileclip_s0';

env.cacheDir = path.join(root, 'build', 'model-cache', 'transformers-js');
env.allowLocalModels = false;

function scoreFromEmbeddings(imageEmbedding, labelEmbeddings, positiveLabels, negativeLabels) {
  const byLabel = new Map(labelEmbeddings.map((item) => [item.label, dot(imageEmbedding, item.embedding)]));
  const positive = Math.max(...positiveLabels.map((label) => byLabel.get(label) ?? Number.NEGATIVE_INFINITY));
  const negative = Math.max(...negativeLabels.map((label) => byLabel.get(label) ?? Number.NEGATIVE_INFINITY));
  return { positive, negative, margin: positive - negative };
}

function orderedUnique(values) {
  const seen = new Set();
  const result = [];
  for (const value of values) {
    const normalized = (value ?? '').trim().replace(/\s+/g, ' ');
    if (!normalized) continue;
    const key = normalized.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      result.push(normalized);
    }
  }
  return result;
}

function containsAny(text, terms) {
  return terms.some((term) => text.includes(term));
}

function semanticQueryPlan(subject) {
  const normalized = subject.trim().replace(/\s+/g, ' ');
  const querySubject = normalized || 'the requested visual content';
  const lowerSubject = querySubject.toLowerCase();
  const wantsScreen = containsAny(lowerSubject, ['screenshot', 'screen capture', 'screen', 'interface', 'gameplay', 'game']);
  const wantsGame = containsAny(lowerSubject, ['game', 'gameplay']);
  const wantsScenery = containsAny(lowerSubject, ['landscape', 'scenery', 'nature', 'outdoor', 'mountain', 'lake', 'beach']);
  const wantsWildlife = containsAny(lowerSubject, ['wildlife', 'wild animal', 'zoo', 'safari']);

  const positivePrompts = [
    querySubject,
    `an image matching ${querySubject}`,
    `a photo containing ${querySubject}`,
    `${querySubject} visible anywhere in the image`,
    `a close-up or background view containing ${querySubject}`,
    wantsScreen
      ? `a screenshot or app interface matching ${querySubject}`
      : `a real camera photo matching ${querySubject}`,
    wantsGame ? 'a video game screenshot' : '',
    wantsGame ? 'a gameplay screen' : '',
    wantsGame ? 'a game interface' : '',
    !wantsWildlife && !wantsScreen ? `a domestic or everyday subject matching ${querySubject}` : '',
  ];

  const negativePrompts = [
    'an unrelated image',
    'a visually similar but wrong image',
    'a different object, animal, or scene',
    'a random photo',
    'a generic image that does not match the request',
    'a blurry or ambiguous image',
    `an image without ${querySubject}`,
    !wantsWildlife ? 'a wildlife animal photo' : '',
    !wantsWildlife ? 'a zoo animal photo' : '',
    !wantsWildlife ? 'a wild predator animal photo' : '',
    wantsScreen ? 'a real camera photo' : 'a screenshot or app interface that does not match the query',
    wantsScreen ? 'a pet or animal photo' : '',
    wantsScreen ? 'a landscape camera photo' : '',
    wantsGame ? 'a generic app screenshot' : '',
    wantsGame ? 'a web app interface screenshot' : '',
    wantsGame ? 'a phone settings screenshot' : '',
  ];

  return {
    positivePrompts: orderedUnique(positivePrompts),
    negativePrompts: orderedUnique(negativePrompts),
    minimumSimilarity: 0.19,
    minimumMargin: wantsGame || wantsScenery ? 0.035 : 0.008,
  };
}

async function ensureFixtureFiles(manifest) {
  const missing = [];
  for (const image of manifest.images) {
    const imagePath = path.join(imageRoot, image.file);
    try {
      const stat = await fs.stat(imagePath);
      if (stat.size <= 0) missing.push(image.file);
    } catch {
      missing.push(image.file);
    }
  }
  if (missing.length > 0) {
    throw new Error(
      `Missing semantic eval images: ${missing.join(', ')}. Run npm run prepare:semantic-eval first.`
    );
  }
}

async function main() {
  const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  await ensureFixtureFiles(manifest);
  await fs.mkdir(resultRoot, { recursive: true });

  console.log(`Loading ${modelId}. First run may download ONNX files into ${env.cacheDir}`);
  const tokenizer = await AutoTokenizer.from_pretrained(modelId);
  const textModel = await CLIPTextModelWithProjection.from_pretrained(modelId);
  const processor = await AutoProcessor.from_pretrained(modelId);
  const visionModel = await CLIPVisionModelWithProjection.from_pretrained(modelId);

  const imageEmbeddings = new Map();
  for (const image of manifest.images) {
    const imagePath = path.join(imageRoot, image.file);
    const rawImage = await RawImage.read(imagePath);
    const imageInputs = await processor(rawImage);
    const { image_embeds } = await visionModel(imageInputs);
    imageEmbeddings.set(image.id, image_embeds.normalize().tolist()[0]);
  }

  const report = {
    generatedAt: new Date().toISOString(),
    model: modelId,
    dataset: path.relative(root, datasetRoot),
    queries: [],
  };
  let failed = false;

  for (const query of manifest.queries) {
    const queryPlan = semanticQueryPlan(query.english);
    const labels = [...queryPlan.positivePrompts, ...queryPlan.negativePrompts];
    const textInputs = tokenizer(labels, { padding: 'max_length', truncation: true });
    const { text_embeds } = await textModel(textInputs);
    const textRows = text_embeds.normalize().tolist();
    const labelEmbeddings = labels.map((label, index) => ({
      label,
      embedding: textRows[index],
    }));
    const scored = [];

    for (const image of manifest.images) {
      const score = scoreFromEmbeddings(
        imageEmbeddings.get(image.id),
        labelEmbeddings,
        queryPlan.positivePrompts,
        queryPlan.negativePrompts
      );
      scored.push({
        id: image.id,
        file: image.file,
        description: image.description,
        positive: score.positive,
        negative: score.negative,
        margin: score.margin,
      });
    }

    scored.sort((a, b) => {
      if (b.margin !== a.margin) return b.margin - a.margin;
      return b.positive - a.positive;
    });

    const accepted = scored.filter(
      (item) => item.positive >= queryPlan.minimumSimilarity && item.margin >= queryPlan.minimumMargin
    );
    const topIds = accepted.slice(0, query.topK).map((item) => item.id);
    const hit = query.expected.some((id) => topIds.includes(id));
    if (!hit) failed = true;

    report.queries.push({
      id: query.id,
      chinese: query.chinese,
      english: query.english,
      expected: query.expected,
      topK: query.topK,
      hit,
      positivePrompts: queryPlan.positivePrompts,
      negativePrompts: queryPlan.negativePrompts,
      minimumSimilarity: queryPlan.minimumSimilarity,
      minimumMargin: queryPlan.minimumMargin,
      ranked: scored,
      accepted,
    });

    console.log(`\n[${hit ? 'PASS' : 'FAIL'}] ${query.chinese} -> ${query.english}`);
    for (const item of scored.slice(0, 5)) {
      const acceptedMark =
        item.positive >= queryPlan.minimumSimilarity && item.margin >= queryPlan.minimumMargin ? '*' : ' ';
      console.log(
        `${acceptedMark} ${item.id.padEnd(18)} margin=${item.margin.toFixed(4)} positive=${item.positive.toFixed(4)} negative=${item.negative.toFixed(4)}`
      );
    }
  }

  const reportPath = path.join(resultRoot, 'mobileclip_eval_report.json');
  await fs.writeFile(reportPath, JSON.stringify(report, null, 2), 'utf8');
  console.log(`\nReport: ${path.relative(root, reportPath)}`);

  if (failed) {
    throw new Error('Semantic image retrieval regression failed.');
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
