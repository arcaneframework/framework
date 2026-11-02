
#arct_begin(OVERWRITE_ARCT)

arct_config_create_or_edit_general(OVERWRITE_ARCT)
arct_config_define_name(NAME "General")
arct_arcane_dataset(PATH "general_dataset.arc")
arct_arccher_mpi(PATH mpi_path)
arct_config_dependency(ADD "config_backwardcpp")
arct_config_create_or_edit_end()

arct_config_create_or_edit_common(NAME_COMMON "config_backwardcpp")
arct_arccher_envvar(ADD "ARCANE_USE_BACKWARDCPP" VALUE "1")
arct_arccher_envvar(ADD "ARCANE_CALLSTACK_HUMAN_READABLE" VALUE "0")
arct_config_create_or_edit_end()



arct_config_create_or_edit_variation(NAME_VARIATION "nb_proc" NAME_VARIANT "2")
arct_config_define_name(NAME "2 procs")
arct_arccher_mpi(NB_PROC 2)
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "nb_proc" NAME_VARIANT "4")
arct_config_define_name(NAME "4 procs")
arct_arccher_mpi(NB_PROC 4)
arct_config_create_or_edit_end()


arct_config_create_or_edit_variation(NAME_VARIATION "nb_thread" NAME_VARIANT "2")
arct_config_define_name(NAME "2 threads")
arct_arcane_option(ADD "S" VALUE "2")
arct_config_create_or_edit_end()


arct_config_create_or_edit_variation(NAME_VARIATION "nb_thread" NAME_VARIANT "4")
arct_config_define_name(NAME "4 threads")
arct_arcane_option(ADD "S" VALUE "4")
arct_config_create_or_edit_end()



arct_config_create_or_edit_variation(NAME_VARIATION "nb_hybrid" NAME_VARIANT "4")
arct_config_define_name(NAME "4 hybrid")
arct_config_variation(ADD "nb_proc" INCLUDE "2")
arct_config_dependency(ADD "nb_thread+2")
arct_config_create_or_edit_end()


arct_config_create_or_edit_variation(NAME_VARIATION "nb_hybrid" NAME_VARIANT "16")
arct_config_define_name(NAME "16 hybrid")
arct_config_dependency(ADD "nb_proc+4")
arct_config_dependency(ADD "nb_thread+4")
arct_config_create_or_edit_end()




arct_config_create_or_edit_variation(NAME_VARIATION "nb_iterations" NAME_VARIANT "10")
arct_config_define_name(NAME "10 itérations")
arct_arcane_option(ADD "MaxIteration" VALUE "10")
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "nb_iterations" NAME_VARIANT "20")
arct_arcane_option(ADD "MaxIteration" VALUE "20")
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "nb_iterations" NAME_VARIANT "30")
arct_arcane_option(ADD "MaxIteration" VALUE "30")
arct_config_create_or_edit_end()




arct_config_create_or_edit_variation(NAME_VARIATION "small_test" NAME_VARIANT "hybrid")
arct_config_variation(ADD "nb_iterations" EXCLUDE "30")
arct_config_variation(ADD "nb_hybrid" EXCLUDE "16")
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "small_test" NAME_VARIANT "proc")
arct_config_variation(ADD "nb_iterations" EXCLUDE "30")
arct_config_variation(ADD "nb_proc")
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "small_test" NAME_VARIANT "thread")
arct_config_variation(ADD "nb_iterations" EXCLUDE "30")
arct_config_variation(ADD "nb_iterations" EXCLUDE "20")
arct_config_variation(ADD "nb_thread")
arct_config_create_or_edit_end()



arct_config_create_or_edit_variation(NAME_VARIATION "big_test" NAME_VARIANT "hybrid")
arct_config_variation(ADD "nb_iterations" INCLUDE "30")
arct_config_variation(ADD "nb_hybrid" EXCLUDE "4")
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "big_test" NAME_VARIANT "proc")
arct_config_variation(ADD "nb_proc")
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "big_test" NAME_VARIANT "thread")
arct_config_variation(ADD "nb_thread")
arct_config_create_or_edit_end()


arct_config_create_or_edit_case(NAME_CASE "mon_test_1")
arct_config_define_name(NAME "Mon Test 1")
arct_config_dependency(ADD "small_test")
arct_arcane_dataset(PATH "truc.arc")
arct_arccher_executable(PATH bin_path)
arct_arccher_envvar(ADD "VARIABLE" VALUE "VAL")
arct_config_create_or_edit_end()

arct_config_create_or_edit_case(NAME_CASE "mon_test_2")
arct_config_define_name(NAME "Mon Test 2")
arct_config_dependency(ADD "big_test")
arct_arcane_dataset(PATH "truc2.arc")
arct_arccher_executable(PATH bin_path)
arct_arccher_envvar(ADD "VARIABLE" VALUE "VAL")
arct_config_create_or_edit_end()

#arct_end()

arct_to_ctest()
