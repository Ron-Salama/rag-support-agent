"""The "forms" the LLM must fill in - one Pydantic model per document type.

Think of each class like a C# class with typed properties + validation attributes:
Pydantic checks the types for us (a date must be a real date, an amount must be a number)
and turns JSON into a Python object (and back).

The `description=` text is not just a comment: it is sent to the model inside the JSON
schema, so it works as part of the prompt. Precise descriptions = fewer extraction errors.

Rules shared by every schema:
  - If a value is not in the document, the model must return null (never guess).
  - Dates are ISO format: YYYY-MM-DD.
  - Amounts are plain numbers exactly as printed (no $ or commas). For financial
    statements we also capture the printed unit ("in thousands") and scale in CODE,
    because arithmetic is something we never ask the LLM to do.
  - `evidence` holds a short verbatim quote for each extracted field, so our code can
    check that every value really appears in the document (a cheap hallucination check).
"""
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    field: str = Field(description="Name of the field this quote supports, e.g. 'total'.")
    quote: str = Field(
        description="A short VERBATIM snippet copied from the document (max ~15 words) "
        "that contains the value. Copy it exactly, do not paraphrase."
    )


# ---------------------------------------------------------------- invoices
class LineItem(BaseModel):
    description: str
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = Field(default=None, description="Line total as printed.")


class Invoice(BaseModel):
    vendor_name: str = Field(description="The company that issued the invoice (the seller).")
    invoice_number: str | None = Field(default=None, description="Invoice number/ID exactly as printed.")
    invoice_date: date | None = None
    due_date: date | None = Field(default=None, description="Payment due date, only if a date is printed.")
    bill_to: str | None = Field(
        default=None,
        description="Name of the customer being billed, as printed under 'Bill To', 'Sold To' or 'Customer name'. "
        "If a company is billed, the company name - not an 'attention' line or a department. If the customer is "
        "an individual, that person's name. NEVER the 'Ship To' / delivery name or address.",
    )
    currency: str = Field(default="USD", description="ISO currency code, e.g. USD.")
    subtotal: float | None = Field(default=None, description="Sum before tax/fees, if printed.")
    tax: float | None = Field(default=None, description="Total tax, if printed.")
    other_charges: float | None = Field(
        default=None,
        description="Shipping/fees as a positive number, discounts/credits as a negative number, if printed.",
    )
    # Two totals on purpose (DECISIONS.md R1): the invoice's own total, and what is owed overall.
    total: float | None = Field(
        default=None,
        description="Total of THIS invoice's charges (subtotal + tax + fees). Do NOT include previous or "
        "unpaid balances carried over from earlier invoices.",
    )
    amount_due: float | None = Field(
        default=None,
        description="The overall amount the customer owes now, if printed (e.g. 'Total Due', 'Balance Due', "
        "'Amount Due'). May include previous unpaid balances. null if no such line is printed.",
    )
    line_items: list[LineItem] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


# ---------------------------------------------------------- financial statements
class FinancialStatement(BaseModel):
    company_name: str
    period_end_date: date | None = Field(
        default=None,
        description="End date of the period you report. If the document shows a three-month "
        "(quarter) column, use the MOST RECENT quarter; otherwise the most recent annual column.",
    )
    period_months: int | None = Field(default=None, description="Length of that period in months: 3, 6, 9 or 12.")
    units: Literal["ones", "thousands", "millions"] = Field(
        description="The unit the table is printed in, e.g. '(in thousands)' -> thousands. 'ones' if not stated."
    )
    total_revenue: float | None = Field(default=None, description="Total revenues, as printed (do not rescale).")
    net_income: float | None = Field(
        default=None,
        description="Net income attributable to the company's common stockholders, as printed. "
        "If that line does not exist, plain net income. Losses are negative.",
    )
    ffo: float | None = Field(
        default=None,
        description="Total NAREIT Funds From Operations (FFO) for the period, as printed. NOT per share. null if absent.",
    )
    total_assets: float | None = Field(default=None, description="Total assets from the balance sheet, as printed. null if no balance sheet.")
    evidence: list[Evidence] = Field(default_factory=list)


# ---------------------------------------------------------------- loan agreements
class LoanTerms(BaseModel):
    borrower_name: str = Field(description="Legal name of the borrower (if several, the first one named).")
    lender_name: str = Field(description="Legal name of the lender (or administrative agent if a syndicate).")
    agreement_date: date | None = Field(
        default=None, description="Date of the agreement or note ('dated as of ...', or the date printed at the top)."
    )
    principal_amount: float | None = Field(default=None, description="Maximum/original loan amount in dollars.")
    interest_rate_type: Literal["fixed", "floating", "unknown"] = Field(
        default="unknown",
        description="'fixed' = a set rate (also a rate that steps up on a fixed schedule); 'floating' = follows a "
        "market index such as SOFR or Prime; 'unknown' = the rate is not stated in THIS document.",
    )
    fixed_rate_percent: float | None = Field(
        default=None,
        description="Fixed annual rate in percent (5.25 means 5.25%). If it steps up on a schedule, the STARTING rate. null if floating.",
    )
    floating_index: str | None = Field(default=None, description="Benchmark for a floating rate, e.g. 'SOFR', 'Prime'. null if fixed.")
    spread_percent: float | None = Field(default=None, description="Margin over the index in percent (3.5 means 3.50%). null if fixed.")
    maturity_date: date | None = None
    collateral_description: str | None = Field(
        default=None, description="Short description of the property/facility securing the loan, if stated."
    )
    evidence: list[Evidence] = Field(default_factory=list)


# ---------------------------------------------------------------- appraisals
class Appraisal(BaseModel):
    property_address: str = Field(
        description="Street address (or name + city/state) of the appraised PROPERTY itself - not the address "
        "of the client the report was prepared for, and not the appraiser's office."
    )
    property_type: Literal[
        "multifamily", "skilled_nursing", "assisted_living", "senior_housing",
        "office", "retail", "industrial", "land", "mixed_use", "other",
    ] = Field(description="Closest category for the building's use (apartments = multifamily). If none fits, 'other'.")
    as_is_market_value: float | None = Field(default=None, description="Final 'as is' market value conclusion in dollars.")
    effective_date: date | None = Field(default=None, description="Effective date of the as-is value (date of value, not report date).")
    appraiser_firm: str | None = None
    building_area_sq_ft: float | None = Field(
        default=None,
        description="Building area in square feet, if stated. If both GROSS and net (rentable) area are given, use the GROSS area.",
    )
    units_or_beds: int | None = Field(default=None, description="Number of apartment units or licensed beds, if stated.")
    cap_rate_percent: float | None = Field(
        default=None,
        description="Capitalization ('cap') rate the appraiser selected and APPLIED in the income approach, in percent "
        "(7.5 means 7.5%). Value = net operating income / cap rate. Not a 'resulting' or implied rate worked "
        "backwards from the final value.",
    )
    net_operating_income: float | None = Field(
        default=None,
        description="Net operating income (NOI): the property's yearly income minus its operating costs, "
        "as used in the income approach, in dollars.",
    )
    evidence: list[Evidence] = Field(default_factory=list)


# doc_type string (as used in data/manifest.csv) -> schema class
SCHEMAS: dict[str, type[BaseModel]] = {
    "invoice": Invoice,
    "financial_statement": FinancialStatement,
    "loan_agreement": LoanTerms,
    "appraisal": Appraisal,
}

# Fields we score against the hand labels (line_items / evidence / free text are not scored).
SCORED_FIELDS: dict[str, list[str]] = {
    "invoice": ["vendor_name", "invoice_number", "invoice_date", "due_date", "bill_to", "subtotal", "tax", "total", "amount_due"],
    "financial_statement": ["company_name", "period_end_date", "period_months", "units", "total_revenue", "net_income", "ffo", "total_assets"],
    "loan_agreement": ["borrower_name", "lender_name", "agreement_date", "principal_amount", "interest_rate_type",
                       "fixed_rate_percent", "floating_index", "spread_percent", "maturity_date"],
    "appraisal": ["property_address", "property_type", "as_is_market_value", "effective_date", "appraiser_firm",
                  "building_area_sq_ft", "units_or_beds", "cap_rate_percent", "net_operating_income"],
}
