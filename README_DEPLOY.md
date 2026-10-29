# MRumah / NestMatch — Vercel deployment

1. Push the **contents of this folder** to the root of a GitHub repository.
2. In Vercel, choose Add New > Project > Import your GitHub repository.
3. Framework Preset: Other; Root Directory: `./`; do not override build/output settings.
4. Under Environment Variables, add `GROQ_API_KEY` with your private Groq key (all environments you deploy to).
5. Deploy. Visit `/`, `/health`, and `/api/listings`; test NestBot on the home page.
6. If you see a Python runtime error, inspect Vercel's Function Logs.

Never commit the original `.env` or `app.env` files; they contain credentials.
The website is served from Flask, and the frontend posts to same-origin `/chat`.
Vercel functions are stateless; long-running tasks and local persistent writes are not supported.
