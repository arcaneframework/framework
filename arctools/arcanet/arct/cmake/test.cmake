
#arct_begin(OVERWRITE_ARCT)

arct_config_create_or_edit_general(OVERWRITE_ARCT)
arct_config_define_name(NAME "General")
arct_arcane_dataset(PATH "general_dataset.arc")
arct_arccher_mpi(PATH mpi_path)
arct_arccher_envvar(ADD "ARCANE_USE_BACKWARDCPP" VALUE "1")
arct_config_create_or_edit_end()


arct_config_create_or_edit_common(NAME_COMMON "4procs")
arct_config_define_name(NAME "4 procs")
arct_arccher_mpi(NB_PROC 4)
arct_config_create_or_edit_end()


arct_config_create_or_edit_common(NAME_COMMON "4threads")
arct_config_define_name(NAME "4 threads")
arct_arcane_option(ADD "S" VALUE "4")
arct_config_create_or_edit_end()


arct_config_create_or_edit_common(NAME_COMMON "16mpithreads")
arct_config_dependency(ADD "4procs")
arct_config_dependency(ADD "4threads")
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "nb_iterations" NAME_VARIANT "10")
arct_arcane_option(ADD "MaxIteration" VALUE "10")
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "nb_iterations" NAME_VARIANT "20")
arct_arcane_option(ADD "MaxIteration" VALUE "20")
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "nb_iterations" NAME_VARIANT "30")
arct_arcane_option(ADD "MaxIteration" VALUE "30")
arct_config_create_or_edit_end()

arct_config_create_or_edit_case(NAME_CASE "mon_test_1")
arct_config_define_name(NAME "Mon Test 1")
arct_config_variation(ADD "nb_iterations" EXCLUDE "10" "20")
arct_config_dependency(AFTER ADD "16mpithreads")
arct_arcane_dataset(PATH "truc.arc")
arct_arcane_option(ADD "MaxIteration" VALUE "3")
arct_arcane_option(ADD "MaxIteration" VALUE "4")
arct_arccher_executable(PATH bin_path)
arct_arccher_envvar(ADD "VARIABLE" VALUE "VAL")
arct_config_create_or_edit_end()

arct_config_create_or_edit_case(NAME_CASE "mon_test_2")
arct_config_define_name(NAME "Mon Test 2")
arct_arcane_dataset(PATH "truc.arc")
arct_arcane_option(ADD "MaxIteration" VALUE "3")
arct_arcane_option(ADD "MaxIteration" VALUE "4")
arct_arccher_executable(PATH bin_path)
arct_arccher_envvar(ADD "VARIABLE" VALUE "VAL")
arct_config_create_or_edit_end()

#arct_end()

arct_to_ctest(ARCCHER_PATH "${CMAKE_BINARY_DIR}/_common/build_all/arcane/arccher/ArcCher")
