
#arct_begin(OVERWRITE_ARCT)

arct_config_create_or_edit_general(OVERWRITE_ARCT)
arct_config_define_name("General")
arct_arcane_set_dataset("general_dataset.arc")
arct_arccher_define_mpi_path(mpi_path)
arct_arccher_envvar(NEW "ARCANE_USE_BACKWARDCPP" VALUE "1")
arct_config_create_or_edit_end()


arct_config_create_or_edit_common(NAME_COMMON "4procs")
arct_config_define_name("4 procs")
arct_arccher_define_mpi_nb_procs(4)
arct_config_create_or_edit_end()


arct_config_create_or_edit_common(NAME_COMMON "4threads")
arct_config_define_name("4 threads")
arct_arcane_option(NEW "S" VALUE "4")
arct_config_create_or_edit_end()


arct_config_create_or_edit_common(NAME_COMMON "16mpithreads")
arct_config_dependency(NEW "4procs")
arct_config_dependency(NEW "4threads")
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "nb_iterations" NAME_VARIANT "10")
arct_arcane_option(NEW "MaxIteration" VALUE "10")
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "nb_iterations" NAME_VARIANT "20")
arct_arcane_option(NEW "MaxIteration" VALUE "20")
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "nb_iterations" NAME_VARIANT "30")
arct_arcane_option(NEW "MaxIteration" VALUE "30")
arct_config_create_or_edit_end()

arct_config_create_or_edit_case(NAME_CASE "mon_test_1")
arct_config_define_name("Mon Test 1")
arct_config_variation(NEW "nb_iterations!10")
arct_config_dependency(AFTER NEW "16mpithreads")
arct_arcane_set_dataset("truc.arc")
arct_arcane_option(NEW "MaxIteration" VALUE "3")
arct_arcane_option(NEW "MaxIteration" VALUE "4")
arct_arccher_define_executable(bin_path)
arct_arccher_envvar(NEW "VARIABLE" VALUE "VAL")
arct_config_create_or_edit_end()

arct_config_create_or_edit_case(NAME_CASE "mon_test_2")
arct_config_define_name("Mon Test 2")
arct_arcane_set_dataset("truc.arc")
arct_arcane_option(NEW "MaxIteration" VALUE "3")
arct_arcane_option(NEW "MaxIteration" VALUE "4")
arct_arccher_define_executable(bin_path)
arct_arccher_envvar(NEW "VARIABLE" VALUE "VAL")
arct_config_create_or_edit_end()

#arct_end()

arct_to_ctest()
