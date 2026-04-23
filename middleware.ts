import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const AUTH_COOKIE_NAME = "khayal_session";

function isPublicAsset(pathname: string): boolean {
  if (pathname.startsWith("/_next")) {
    return true;
  }
  if (pathname.startsWith("/api")) {
    return true;
  }
  if (pathname.includes(".")) {
    return true;
  }
  return false;
}

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  if (isPublicAsset(pathname)) {
    return NextResponse.next();
  }

  const token = request.cookies.get(AUTH_COOKIE_NAME)?.value;
  const hasSession = Boolean(token);

  if (!hasSession && pathname !== "/auth") {
    return NextResponse.redirect(new URL("/auth", request.url));
  }

  if (hasSession && pathname === "/auth") {
    return NextResponse.redirect(new URL("/", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/:path*"]
};
