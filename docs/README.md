# Viewing the docs

The HTML guide lives in `index.html` in this directory.

## Start the server (local machine)

From the repo root:

```bash
cp .env.example .env
docker compose up docs-server
```

Open [http://localhost:8080](http://localhost:8080).

`/` and `/index.html` both serve `docs/index.html`.

## Port

Default port is `8080`. Change it in `.env`:

```bash
DOCS_PORT=8080
```

## Stop the server

```bash
docker compose stop docs-server
```

No image build is required. On first run, Docker pulls the nginx image.
