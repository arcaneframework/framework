# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

loadLanguage(NAME axl PATH ${BUILD_SYSTEM_PATH}/languages)

git add if(USE_ARCANE_V3)
  set(AXL2CC "${AXLSTAR_AXL2CC}")
  logStatus("AXL TOOLS : AXL2CC = ${AXLSTAR_AXL2CC} AXL2CCT4 = ${AXL2CCT4}")
endif()

