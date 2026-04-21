const PROFILE_NAME_PATTERN = /^[a-zA-Z0-9 _-]{2,50}$/;

export function isEdfFileName(fileName: string): boolean {
  return fileName.toLowerCase().endsWith(".edf");
}

export function isPtFilePath(filePath: string): boolean {
  return filePath.toLowerCase().endsWith(".pt");
}

export function isSafeProfileName(name: string): boolean {
  return PROFILE_NAME_PATTERN.test(name.trim());
}

export function assert(condition: unknown, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}
