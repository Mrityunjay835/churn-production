from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class ChurnRequest(BaseModel):
    tenure: int = Field(
        ge = 0,
        le = 72,
        description = "Months customer has been with company",
        examples = 12
    )

    MonthlyCharges : float = Field(
        ge = 0,
        description = "Monthly bill amount in USD",
        examples = 65.50
    )
    TotalCharges: float = Field(
            ge=0,
            description="Total amount charged to date",
            example=786.0
        )
    Contract: str = Field(
            description="Contract type",
            example="Month-to-month"
        )
    PaymentMethod: str = Field(
            description="Payment method",
            example="Electronic check"
        )
    PaperlessBilling: str = Field(
            description="Whether customer uses paperless billing",
            example="Yes"
        )

    # Demographics
    gender: str = Field(example="Male")
    SeniorCitizen: int = Field(ge=0, le=1, example=0)
    Partner: str = Field(example="Yes")
    Dependents: str = Field(example="No")

    # Phone services
    PhoneService: str = Field(example="Yes")
    MultipleLines: str = Field(example="No")

    # Internet services
    InternetService: str = Field(example="Fiber optic")
    OnlineSecurity: str = Field(example="No")
    OnlineBackup: str = Field(example="Yes")
    DeviceProtection: str = Field(example="No")
    TechSupport: str = Field(example="No")
    StreamingTV: str = Field(example="No")
    StreamingMovies: str = Field(example="No")




class ChurnResponse(BaseModel):
    """
    Output schema for churn prediction.
    Always return structured, typed responses — never raw dicts.
    """
    model_config = ConfigDict(protected_namespaces=())

    churn_probability: float = Field(
        description="Probability of churn between 0 and 1",
        example=0.73
    )
    churn_prediction: bool = Field(
        description="True if predicted to churn",
        example=True
    )
    risk_level: str = Field(
        description="Human readable risk level",
        example="HIGH"
    )
    threshold_used: float = Field(
        description="Decision threshold applied",
        example=0.33
    )
    model_version: str = Field(
        description="MLflow model version used",
        example="2"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    model_config = ConfigDict(protected_namespaces=())
    status: str = Field(example="ok")
    model_loaded: bool = Field(example=True)
    model_version: str = Field(example="2")
    threshold: float = Field(example=0.33)


class ErrorResponse(BaseModel):
    """Standard error response."""
    model_config = ConfigDict(protected_namespaces=())
    error: str
    detail: Optional[str] = None
