import pytest

import tune_the_model as ttm


@pytest.fixture(scope="module")
def public_models(configured_tune_the_model):
    return ttm.TuneTheModel.public_models()


@pytest.fixture
def public_model(public_models, request):
    return next((m for m in public_models if m.name == request.param), None)


@pytest.mark.parametrize(
    "public_model",
    [
        "massive_intent_classification",
        "massive_slot_detection_gen",
        "amazon_description_by_title_en_ru_ar",
        "amazon_title_by_description_en_ru_ar",
        "factcheck_classifier_en_ru_ar",
        "factcheck_random_fact_en_ru_ar",
        "cleanweb_en_ru_ar",
    ],
    indirect=True
)
def test_public_model_lookup(public_model):
    assert public_model is not None


@pytest.mark.parametrize("public_model", ["massive_intent_classification"], indirect=True)
def test_intent_classifier(public_model):
    result = public_model.classify("test")

    assert len(result) > 0


@pytest.mark.parametrize("public_model", ["massive_slot_detection_gen"], indirect=True)
def test_slot_detection_generator(public_model):
    result = public_model.generate("Remind me to call Thomas tomorrow")[0]

    assert "Remind me" in result
    assert "Thomas" in result


@pytest.mark.parametrize("public_model", ["amazon_description_by_title_en_ru_ar"], indirect=True)
def test_desc_by_title_generator(public_model):
    result = public_model.generate("Title")

    assert len(result) > 0


@pytest.mark.parametrize("public_model", ["amazon_title_by_description_en_ru_ar"], indirect=True)
def test_title_by_desc_generator(public_model):
    result = public_model.generate("Description")

    assert len(result) > 0


@pytest.mark.parametrize("public_model", ["factcheck_classifier_en_ru_ar"], indirect=True)
def test_factcheck_classifier(public_model):
    result = public_model.classify("Test")

    assert len(result) > 0


@pytest.mark.parametrize("public_model", ["factcheck_random_fact_en_ru_ar"], indirect=True)
def test_factcheck_random_fact_generator(public_model):
    result = public_model.generate("Test")

    assert len(result) > 0


@pytest.mark.parametrize("public_model", ["cleanweb_en_ru_ar"], indirect=True)
def test_cleanweb_classifier(public_model):
    result = public_model.classify("Test")

    assert len(result) > 0
