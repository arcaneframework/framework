cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(MiniNbody LANGUAGES C CXX)

find_package(Arcane REQUIRED)
add_executable(MiniNbody main.cc)
arcane_add_arcane_libraries_to_target(MiniNbody)

# Pour ajouter le support des accélérateurs
arcane_accelerator_enable()
arcane_accelerator_add_source_files(main.cc)
arcane_accelerator_add_to_target(MiniNbody)

# Recopie le jeu de données
add_test(NAME mini_nbody COMMAND MiniNbody)
