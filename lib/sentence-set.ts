import sentenceCatalog from "@/data/sentence_catalog.json";
import labels from "@/data/labels.json";
import type { LabelItem, SentenceCatalogItem } from "@/lib/types";

export const SENTENCE_CATALOG = sentenceCatalog.sentence_catalog as SentenceCatalogItem[];
export const LABELS = labels.labels as LabelItem[];

export const SENTENCE_COUNT = SENTENCE_CATALOG.length;
export const VOCABULARY_SIZE = LABELS.length;

export function getSentenceById(sentenceId: string): SentenceCatalogItem | undefined {
  return SENTENCE_CATALOG.find((item) => item.sentence_id === sentenceId);
}

export function tokenForLabel(labelId: number): string {
  return LABELS.find((item) => item.id === labelId)?.word ?? "";
}
