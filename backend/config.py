import os
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Required for LLM
    google_api_key: SecretStr = Field(..., alias="GOOGLE_API_KEY")
    
    # Required for Vector Search (Vertex AI)
    gcp_project_id: str = Field(..., alias="GCP_PROJECT_ID")
    index_endpoint_id_full: str = Field(..., alias="INDEX_ENDPOINT_ID_FULL")
    deployed_index_id: str = Field(..., alias="DEPLOYED_INDEX_ID")
    
    # Optional with Defaults
    gcp_region: str = Field("europe-west4", alias="GCP_REGION")
    bq_dataset_id: str = Field("ks_metadata", alias="BQ_DATASET_ID")
    bq_table_id: str = Field("docstore", alias="BQ_TABLE_ID")

    # Load from .env file
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

def validate_config():
    """Validates presence of critical env vars at startup."""
    try:
        settings = Settings()
        print("✅ Environment variables validated successfully.")
        return settings
    except Exception as e:
        print("\n❌ CRITICAL ERROR: Missing or Invalid Environment Variables")
        print("Please check your .env file or system environment.")
        print(f"Details: {e}")
        # Exit the process to prevent 'zombie' deployment
        import sys
        sys.exit(1)