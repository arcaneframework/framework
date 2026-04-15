
set(ARCANE_TEST_CASEPATH ${TEST_PATH}/geometry)

arcane_add_test(geometry testGeom.arc)
arcane_add_test(geometry_applyoperation_v2 testGeom.arc -We,ARCANE_DEBUG_APPLYOPERATION,1)
arcane_add_test(geometry_applyoperation_v1 testGeom.arc -We,ARCANE_APPLYOPERATION_VERSION,1 -We,ARCANE_DEBUG_APPLYOPERATION,1)

#############
# GEOMETRIC #
#############
arcane_add_test_sequential(geometric1 testGeometric-1.arc)
