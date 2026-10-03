"""API HTTP para inferência do modelo de câncer de mama."""

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

CAMINHO_MODELO = Path(__file__).with_name("model.pickle")

with CAMINHO_MODELO.open("rb") as arquivo:
    modelo = pickle.load(arquivo)

COLUNAS_ESPERADAS = list(modelo.feature_names_in_)
CLASSES = list(modelo.classes_)
INDICE_MALIGNO = CLASSES.index(1)


def criar_api() -> FastAPI:
    """Cria e configura a aplicação, sem iniciar o servidor HTTP."""
    app = FastAPI(
        title="API de classificação de câncer de mama",
        version="1.0.0",
        description="Classifica uma observação como benigna ou maligna.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Indica que o modelo foi carregado com sucesso."""
        return {"status": "ok"}

    @app.post("/predict")
    def predict(dados: dict[str, Any]) -> dict[str, float | int | str]:
        """Recebe as colunas do modelo e retorna a classificação da observação."""
        colunas_recebidas = set(dados)
        faltantes = sorted(set(COLUNAS_ESPERADAS) - colunas_recebidas)
        desconhecidas = sorted(colunas_recebidas - set(COLUNAS_ESPERADAS))

        if faltantes or desconhecidas:
            detalhe: dict[str, list[str]] = {}
            if faltantes:
                detalhe["colunas_faltantes"] = faltantes
            if desconhecidas:
                detalhe["colunas_desconhecidas"] = desconhecidas
            raise HTTPException(status_code=422, detail=detalhe)

        entrada = pd.DataFrame([dados], columns=COLUNAS_ESPERADAS)
        try:
            previsao = int(modelo.predict(entrada)[0])
            probabilidade_maligno = float(
                modelo.predict_proba(entrada)[0][INDICE_MALIGNO]
            )
        except (TypeError, ValueError) as erro:
            raise HTTPException(
                status_code=422,
                detail="Os valores informados não são compatíveis com as colunas do modelo.",
            ) from erro

        return {
            "classe": previsao,
            "diagnostico": "maligno" if previsao == 1 else "benigno",
            "probabilidade_maligno": probabilidade_maligno,
        }

    return app


app: FastAPI = criar_api()
