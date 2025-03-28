include(CMakeParseArguments)

macro(load_bench)

    set(options PARALLEL)
    set(oneValueArgs BENCH NAME)
    set(multiValueArgs)

    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (ARGS_UNPARSED_ARGUMENTS)
        logFatalError("unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
    endif ()

    get_property(benchs GLOBAL PROPERTY ${PROJECT_NAME}_BENCHS)

    list(APPEND benchs ${ARGS_BENCH})
    list(REMOVE_DUPLICATES benchs)

    set_property(GLOBAL PROPERTY ${PROJECT_NAME}_BENCHS ${benchs})

    # NB: APPEND ne marche que pour des listes existantes
    if (NOT ARGS_PARALLEL)
        get_property(tests GLOBAL PROPERTY ${ARGS_BENCH}_TESTS)
        list(APPEND tests ${ARGS_NAME})
        set_property(GLOBAL PROPERTY ${ARGS_BENCH}_TESTS ${tests})
    else ()
        get_property(tests GLOBAL PROPERTY ${ARGS_BENCH}_PARALLEL_TESTS)
        list(APPEND tests ${ARGS_NAME})
        set_property(GLOBAL PROPERTY ${ARGS_BENCH}_PARALLEL_TESTS ${tests})
    endif ()

endmacro()

macro(alien_test)

  set(options PARALLEL_ONLY UNIQUE_OUTPUT_DIR)
    set(oneValueArgs BENCH NAME COMMAND WORKING_DIRECTORY)
    set(multiValueArgs OPTIONS PROCS)

    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (ARGS_UNPARSED_ARGUMENTS)
        logFatalError("unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
    endif ()

    if (NOT ARGS_BENCH)
        logFatalError("a test should be part of a bench : BENCH not defined")
    endif ()

    if (NOT ARGS_NAME)
        logFatalError("a test should have a name : NAME not defined")
    endif ()

    if (NOT ARGS_COMMAND)
        logFatalError("a test should be related to an executable : COMMAND not defined")
    endif ()

    if (NOT ARGS_PARALLEL_ONLY)

        if (NOT ARGS_WORKING_DIRECTORY)
            add_test(
                    NAME alien.${ARGS_BENCH}.${ARGS_NAME}
                    COMMAND ${ARGS_COMMAND}
                    ${ARGS_OPTIONS}
            )
        else ()
            if(ARGS_UNIQUE_OUTPUT_DIR)
              set(ALIEN_TEST_OUTDIR ${CMAKE_BINARY_DIR}/${ARGS_WORKING_DIRECTORY}/alien.${ARGS_BENCH}.${ARGS_NAME})
              set(ALIEN_TEST_PARAM_OUTDIR -A,OutputDirectory=${ALIEN_TEST_OUTDIR})
              file(MAKE_DIRECTORY ${ALIEN_TEST_OUTDIR})
            else ()
              set(ALIEN_TEST_PARAM_OUTDIR "")
            endif()
            add_test(
                    NAME alien.${ARGS_BENCH}.${ARGS_NAME}
                    COMMAND ${ARGS_COMMAND} ${ALIEN_TEST_PARAM_OUTDIR}
                    ${ARGS_OPTIONS}
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/${ARGS_WORKING_DIRECTORY}
            )
        endif ()

        if (WIN32)
            # ajout de bin/sys_dlls au PATH
            set(PATH ${CMAKE_BINARY_DIR}/bin/sys_dlls)
            list(APPEND PATH $ENV{PATH})
            string(REPLACE ";" "\\;" PATH "${PATH}")
            set_property(TEST alien.${ARGS_BENCH}.${ARGS_NAME} APPEND PROPERTY ENVIRONMENT "PATH=${PATH}")
        endif ()

        load_bench(BENCH ${ARGS_BENCH} NAME ${ARGS_NAME})

    endif ()

    if (ARGS_PROCS OR ARGS_PARALLEL_ONLY)

        if (NOT TARGET mpi)
            logStatus("parallel test ${ARGS_NAME} can't be defined : mpi not available")
            return()
        endif ()

        if (NOT ARGS_PROCS)
            set(ARGS_PROCS 1)
        endif ()

        foreach (mpi ${ARGS_PROCS})

            if (NOT ARGS_WORKING_DIRECTORY)
                add_test(
                        NAME alien.${ARGS_BENCH}.${ARGS_NAME}.mpi-${mpi}
                        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${mpi} ${MPIEXEC_PREFLAGS}$<TARGET_FILE:${ARGS_COMMAND}> ${MPIEXEC_POSTFLAGS}
                        ${ARGS_OPTIONS}
                )
            else ()
                if(ARGS_UNIQUE_OUTPUT_DIR)
                  set(ALIEN_TEST_OUTDIR ${CMAKE_BINARY_DIR}/${ARGS_WORKING_DIRECTORY}/alien.${ARGS_BENCH}.${ARGS_NAME}.mpi-${mpi})
                  set(ALIEN_TEST_PARAM_OUTDIR -A,OutputDirectory=${ALIEN_TEST_OUTDIR})
                  file(MAKE_DIRECTORY ${ALIEN_TEST_OUTDIR})
                else ()
                  set(ALIEN_TEST_PARAM_OUTDIR "")
                endif ()
                add_test(
                        NAME alien.${ARGS_BENCH}.${ARGS_NAME}.mpi-${mpi}
                        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${mpi} ${MPIEXEC_PREFLAGS}$<TARGET_FILE:${ARGS_COMMAND}> ${MPIEXEC_POSTFLAGS} ${ALIEN_TEST_PARAM_OUTDIR}
                        ${ARGS_OPTIONS}
                        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/${ARGS_WORKING_DIRECTORY}
                )
            endif ()

            if (WIN32)
                # ajout de bin/sys_dlls au PATH
                set(PATH ${CMAKE_BINARY_DIR}/bin/sys_dlls)
                list(APPEND PATH $ENV{PATH})
                string(REPLACE ";" "\\;" PATH "${PATH}")
                set_property(TEST alien.${ARGS_BENCH}.${ARGS_NAME}.mpi-${mpi} APPEND PROPERTY ENVIRONMENT "PATH=${PATH}")
            endif ()

            load_bench(BENCH ${ARGS_BENCH} NAME ${ARGS_NAME} PARALLEL)

        endforeach ()

    endif ()

endmacro()

function(print_bench_informations)

    get_property(benchs GLOBAL PROPERTY ${PROJECT_NAME}_BENCHS)

    list(LENGTH benchs nb_benchs)

    logStatus("Load ${nb_benchs} test bench(s)")

    foreach (bench ${benchs})

        logStatus(" ** Bench ${BoldMagenta}${bench}${ColourReset}")

        get_property(bench_seq GLOBAL PROPERTY ${bench}_TESTS)

        list(LENGTH bench_seq nb_seq_tests)

        logStatus("  * sequential : ${nb_seq_tests} test(s)")

        get_property(bench_par GLOBAL PROPERTY ${bench}_PARALLEL_TESTS)

        list(LENGTH bench_par nb_mpi_tests)

        logStatus("  *   parallel : ${nb_mpi_tests} test(s)")

    endforeach ()

endfunction()
