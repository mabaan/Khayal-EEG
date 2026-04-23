import { randomBytes, scryptSync, timingSafeEqual } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { ensureDir, paths } from "@/lib/local-paths";

export const AUTH_COOKIE_NAME = "khayal_session";

interface AuthUser {
  id: string;
  username: string;
  username_key: string;
  password_hash: string;
  salt: string;
  created_at: string;
}

interface AuthSession {
  token: string;
  user_id: string;
  username: string;
  created_at: string;
  expires_at: string;
}

interface AuthDb {
  users: AuthUser[];
  sessions: AuthSession[];
}

const AUTH_DIR = path.join(paths.storageRoot, "auth");
const AUTH_FILE = path.join(AUTH_DIR, "auth_db.json");

function nowIso(): string {
  return new Date().toISOString();
}

function usernameKey(value: string): string {
  return value.trim().toLowerCase();
}

function sessionExpiry(days = 30): string {
  const next = new Date();
  next.setDate(next.getDate() + days);
  return next.toISOString();
}

function hashPassword(password: string, salt: string): string {
  return scryptSync(password, salt, 64).toString("hex");
}

async function readAuthDb(): Promise<AuthDb> {
  await ensureDir(AUTH_DIR);

  try {
    const raw = await fs.readFile(AUTH_FILE, "utf8");
    const parsed = JSON.parse(raw) as Partial<AuthDb>;
    return {
      users: Array.isArray(parsed.users) ? parsed.users : [],
      sessions: Array.isArray(parsed.sessions) ? parsed.sessions : []
    };
  } catch {
    const initial: AuthDb = { users: [], sessions: [] };
    await writeAuthDb(initial);
    return initial;
  }
}

async function writeAuthDb(db: AuthDb): Promise<void> {
  await ensureDir(AUTH_DIR);
  await fs.writeFile(AUTH_FILE, `${JSON.stringify(db, null, 2)}\n`, "utf8");
}

function assertCredentials(username: string, password: string): void {
  if (username.trim().length < 3) {
    throw new Error("Username must be at least 3 characters.");
  }
  if (password.length < 6) {
    throw new Error("Password must be at least 6 characters.");
  }
}

export async function registerUser(username: string, password: string): Promise<{ userId: string; username: string }> {
  assertCredentials(username, password);

  const db = await readAuthDb();
  const key = usernameKey(username);

  if (db.users.some((user) => user.username_key === key)) {
    throw new Error("Username already exists.");
  }

  const salt = randomBytes(16).toString("hex");
  const user: AuthUser = {
    id: `user-${Date.now()}`,
    username: username.trim(),
    username_key: key,
    password_hash: hashPassword(password, salt),
    salt,
    created_at: nowIso()
  };

  db.users.push(user);
  await writeAuthDb(db);

  return { userId: user.id, username: user.username };
}

export async function authenticateUser(username: string, password: string): Promise<{ userId: string; username: string }> {
  assertCredentials(username, password);

  const db = await readAuthDb();
  const key = usernameKey(username);
  const user = db.users.find((item) => item.username_key === key);

  if (!user) {
    throw new Error("Invalid username or password.");
  }

  const expected = Buffer.from(user.password_hash, "hex");
  const actual = Buffer.from(hashPassword(password, user.salt), "hex");

  if (expected.length !== actual.length || !timingSafeEqual(expected, actual)) {
    throw new Error("Invalid username or password.");
  }

  return { userId: user.id, username: user.username };
}

export async function createSession(userId: string, username: string): Promise<string> {
  const db = await readAuthDb();
  const token = randomBytes(32).toString("hex");

  db.sessions = db.sessions.filter((session) => new Date(session.expires_at).getTime() > Date.now());
  db.sessions.push({
    token,
    user_id: userId,
    username,
    created_at: nowIso(),
    expires_at: sessionExpiry(30)
  });

  await writeAuthDb(db);
  return token;
}

export async function getSession(token: string): Promise<AuthSession | null> {
  if (!token) {
    return null;
  }

  const db = await readAuthDb();
  const session = db.sessions.find((item) => item.token === token);

  if (!session) {
    return null;
  }

  if (new Date(session.expires_at).getTime() <= Date.now()) {
    db.sessions = db.sessions.filter((item) => item.token !== token);
    await writeAuthDb(db);
    return null;
  }

  return session;
}

export async function invalidateSession(token: string): Promise<void> {
  if (!token) {
    return;
  }

  const db = await readAuthDb();
  const next = db.sessions.filter((session) => session.token !== token);

  if (next.length !== db.sessions.length) {
    db.sessions = next;
    await writeAuthDb(db);
  }
}
