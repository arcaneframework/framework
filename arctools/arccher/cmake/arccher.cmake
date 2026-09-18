


macro(arcanet_begin)
  if(NOT EXISTS ${CMAKE_BINARY_DIR}/testlist.arct)
    set(testlistjson "{}")
    string(JSON testlistjson SET ${testlistjson} "versions" "{}")
    string(JSON testlistjson SET ${testlistjson} "versions" "_" "0")
    string(JSON testlistjson SET ${testlistjson} "versions" "arcane" "0")
    string(JSON testlistjson SET ${testlistjson} "versions" "arccher" "0")
    file(WRITE ${CMAKE_BINARY_DIR}/testlist.arct ${testlistjson})
  endif ()
endmacro()


# arcanet_begin()

# arcanet_create_or_edit_general()
# arcanet_define_name("General")
# arcanet_arcane_set_dataset("general_dataset.arc")
# arcanet_arccher_define_mpi_path(mpi_path)
# arcanet_arccher_add_envvar("ARCANE_USE_BACKWARDCPP" "1")
# arcanet_create_or_edit_end()

# arcanet_create_or_edit_common("4procs")
# arcanet_define_name("4 procs")
# arcanet_arccher_define_mpi_nb_procs(4)
# arcanet_create_or_edit_end()

# arcanet_create_or_edit_case("mon_test_1")
# arcanet_define_name("Mon Test 1")
# arcanet_arcane_set_dataset("truc.arc")
# arcanet_arcane_add_option("MaxIteration" "3")
# arcanet_arccher_define_executable(bin_path)
# arcanet_arccher_add_envvar("VARIABLE" "VALUE")
# arcanet_create_or_edit_end()

# arcanet_end()
