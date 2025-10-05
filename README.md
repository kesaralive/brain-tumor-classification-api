# FastAPI Project - Local Development

This guide explains how to run the FastAPI project locally using **Poetry**.

## Prerequisites

1. **Python 3.11+** installed
2. **Poetry** - Python dependency manager

### Installing Poetry

If you don’t have Poetry installed, you can install it with:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Make sure to add Poetry to your PATH:

```
export PATH="$HOME/.local/bin:$PATH"
```

Verify installation:
```
poetry --version
```

## Running the Project Locally

Clone the repository:
```
git clone <repository-url>
cd <repository-folder>
```

Install dependencies using Poetry:
```
poetry install
```

Run the development server:
```
poetry run server
```

By default, the FastAPI server will start on `http://127.0.0.1:8081` (or the port defined in your project settings).