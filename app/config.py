from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    chroma_path: str = "./chroma_db"
    top_k: int = 5
    rate_limit: str = "20/minute"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
