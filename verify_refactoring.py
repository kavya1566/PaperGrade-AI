"""Smoke test — verifies all services, the inference engine, and the digital twin are wired correctly."""

import pandas as pd
from ml.pipeline import FeatureExtractor
from simulator.assessment import classify_transition_outcome, compute_failure_severity, summarize_transition
from simulator.engine import DigitalTwin
from services.prediction_service import PredictionService
from services.explanation_service import ExplanationService
from services.recommendation_service import RecommendationService
from services.history_service import HistoricalIntelligenceService
from services.decision_engine import DecisionEngine, DecisionContext
from ml.inference import InferenceEngine


def main():
    print("PAPERGRADE AI ARCHITECTURAL REFINEMENT VERIFICATION")
    print("=" * 60)
    print()

    # 1. Feature Extraction & Leakage Verification
    df = pd.read_csv("simulator/historian.csv")
    transition_df = df[df["transition_id"] == df["transition_id"].iloc[0]].head(10).reset_index(drop=True)

    extractor = FeatureExtractor(transition_df)
    features = extractor.extract()

    leakage_features = {
        "early_off_spec_count": "early_off_spec_count" in features,
        "early_off_spec_fraction": "early_off_spec_fraction" in features,
        "first_off_spec_minute": "first_off_spec_minute" in features,
        "peak_deviation_seen": "peak_deviation_seen" in features,
    }

    print("[1] Feature Extraction Leakage Audit:")
    for feat, present in leakage_features.items():
        status = "PRESENT (ERROR!)" if present else "REMOVED (OK)"
        print(f"  {feat}: {status}")

    # 2. Process Assessment Layer Verification
    print("\n[2] Process Assessment Layer Verification (simulator/assessment.py):")
    sample_rows = transition_df.to_dict(orient="records")
    summary = summarize_transition(sample_rows)
    outcome = classify_transition_outcome(bw_off_spec_minutes=0, bw_peak_dev=1.2, recovery_minutes=1)
    severity = compute_failure_severity("STEAM_VALVE_LAG", 12)

    print(f"  Summary outcome evaluation: {summary.get('transition_outcome')}")
    print(f"  Quality outcome classification: {outcome} (Expected: SUCCESS)")
    print(f"  Failure severity evaluation: {severity} (Expected: HIGH)")

    # 3. Digital Twin Simulation Layer Verification
    print("\n[3] Digital Twin Simulation Layer Verification (simulator/engine.py):")
    resim_df, resim_metrics = DigitalTwin.simulate(grade_from="G20", grade_to="G70", duration_minutes=30)
    print(f"  DigitalTwin simulation produced {len(resim_df)} timesteps, outcome: {resim_metrics['transition_outcome']}")

    # 4. Decision Intelligence Services & DecisionEngine Verification
    print("\n[4] Decision Intelligence Services & DecisionEngine Verification:")
    decision_engine = DecisionEngine()
    context = decision_engine.evaluate_decision_package(
        df_row=pd.DataFrame([features]),
        grade_from="G20",
        grade_to="G70",
        failure_type="STEAM_VALVE_LAG",
    )

    is_valid_context = isinstance(context, DecisionContext)
    has_pred = "risk_probability" in context.prediction
    has_exp = "physical_interpretation" in context.explanation
    has_rec = "actuator_deltas" in context.recommendation
    has_val = "validation_df" in context.validation
    has_hist = len(context.historical_evidence) > 0

    print(f"  DecisionContext Created: {is_valid_context}")
    print(f"  PredictionService Output: {'OK' if has_pred else 'FAIL'}")
    print(f"  ExplanationService Output: {'OK' if has_exp else 'FAIL'}")
    print(f"  RecommendationService Output: {'OK' if has_rec else 'FAIL'}")
    print(f"  Digital Twin Validation Output: {'OK' if has_val else 'FAIL'}")
    print(f"  HistoricalIntelligence Output ({len(context.historical_evidence)} matches): {'OK' if has_hist else 'FAIL'}")

    # 5. Inference Engine Facade Verification
    print("\n[5] InferenceEngine Facade Verification (ml/inference.py):")
    ie = InferenceEngine()
    health = ie.health_check()
    pred_res = ie.predict(pd.DataFrame([features]))
    print(f"  Model Loaded: {health['model_loaded']} | Feature Count: {health['feature_count']}")
    print(f"  Prediction Probability: {pred_res.get('risk_probability')} | Prediction: {pred_res.get('prediction')}")

    # 6. Dictionary & DataFrame Dual-Input API Compatibility Verification
    print("\n[6] Dictionary & DataFrame Dual-Input API Compatibility Verification:")
    dict_features = dict(features)

    # Test PredictionService with dict
    ps_dict_prep = decision_engine.prediction_service.preprocess_features(dict_features)
    ps_dict_pred = decision_engine.prediction_service.predict(dict_features)
    has_ps_dict = isinstance(ps_dict_prep, pd.DataFrame) and "risk_probability" in ps_dict_pred

    # Test ExplanationService with dict
    exp_dict = decision_engine.explanation_service.explain_instance(dict_features, decision_engine.prediction_service)
    has_exp_dict = "physical_interpretation" in exp_dict

    # Test DecisionEngine with dict
    context_dict = decision_engine.evaluate_decision_package(
        df_row=dict_features,
        grade_from="G20",
        grade_to="G70",
        failure_type="STEAM_VALVE_LAG",
    )
    has_de_dict = isinstance(context_dict, DecisionContext) and "risk_probability" in context_dict.prediction

    # Test InferenceEngine with dict
    ie_dict_pred = ie.predict(dict_features)
    has_ie_dict = "risk_probability" in ie_dict_pred

    print(f"  PredictionService(dict input): {'OK' if has_ps_dict else 'FAIL'}")
    print(f"  ExplanationService(dict input): {'OK' if has_exp_dict else 'FAIL'}")
    print(f"  DecisionEngine(dict input): {'OK' if has_de_dict else 'FAIL'}")
    print(f"  InferenceEngine(dict input): {'OK' if has_ie_dict else 'FAIL'}")

    dict_passed = has_ps_dict and has_exp_dict and has_de_dict and has_ie_dict

    # Final Audit Check
    errors = sum(leakage_features.values())
    all_passed = (errors == 0) and is_valid_context and has_pred and has_exp and has_rec and has_val and has_hist and dict_passed

    print("\n" + "=" * 60)
    if all_passed:
        print("[PASS] ALL ARCHITECTURAL REFINEMENT VERIFICATIONS PASSED SUCCESSFULLY!")
        return 0
    else:
        print("[FAIL] VERIFICATION FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())
