import { NextRequest, NextResponse } from "next/server";
import { AUTH_COOKIE_NAME, invalidateSession } from "@/lib/auth-store";

export async function POST(request: NextRequest) {
  const token = request.cookies.get(AUTH_COOKIE_NAME)?.value ?? "";
  await invalidateSession(token);

  const response = NextResponse.json({ status: "ok" });
  response.cookies.set(AUTH_COOKIE_NAME, "", {
    httpOnly: true,
    sameSite: "lax",
    secure: false,
    path: "/",
    maxAge: 0
  });

  return response;
}
