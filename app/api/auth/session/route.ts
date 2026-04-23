import { NextRequest, NextResponse } from "next/server";
import { AUTH_COOKIE_NAME, getSession } from "@/lib/auth-store";

export async function GET(request: NextRequest) {
  const token = request.cookies.get(AUTH_COOKIE_NAME)?.value ?? "";
  const session = await getSession(token);

  if (!session) {
    return NextResponse.json({ authenticated: false, user: null });
  }

  return NextResponse.json({
    authenticated: true,
    user: {
      id: session.user_id,
      username: session.username
    }
  });
}
