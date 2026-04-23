import { NextRequest, NextResponse } from "next/server";
import { AUTH_COOKIE_NAME, createSession, registerUser } from "@/lib/auth-store";

export async function POST(request: NextRequest) {
  try {
    const payload = (await request.json()) as { username?: string; password?: string };
    const username = payload.username?.trim() ?? "";
    const password = payload.password ?? "";

    const user = await registerUser(username, password);
    const token = await createSession(user.userId, user.username);

    const response = NextResponse.json({ status: "ok", username: user.username });
    response.cookies.set(AUTH_COOKIE_NAME, token, {
      httpOnly: true,
      sameSite: "lax",
      secure: false,
      path: "/",
      maxAge: 60 * 60 * 24 * 30
    });

    return response;
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Registration failed." },
      { status: 400 }
    );
  }
}
