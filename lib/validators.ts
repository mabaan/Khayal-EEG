const PROFILE_NAME_PATTERN = /^[a-zA-Z0-9 _-]{2,50}$/;

export function isEdfFileName(fileName: string): boolean {
  return fileName.toLowerCase().endsWith(".edf");
}

export function isCsvFileName(fileName: string): boolean {
  return fileName.toLowerCase().endsWith(".csv");
}

export function isModelFileName(fileName: string): boolean {
  const lower = fileName.toLowerCase();
  return lower.endsWith(".pt") || lower.endsWith(".pth");
}

export function isPtFilePath(filePath: string): boolean {
  const lower = filePath.toLowerCase();
  return lower.endsWith(".pt") || lower.endsWith(".pth");
}

export function isSafeProfileName(name: string): boolean {
  return PROFILE_NAME_PATTERN.test(name.trim());
}

export function assert(condition: unknown, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}
