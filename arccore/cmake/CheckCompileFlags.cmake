# ----------------------------------------------------------------------------
# Ajoute des flags de compilation pour les avertissements s'ils existent

include(CheckCXXCompilerFlag)

if (NOT WIN32)
  # NOTE: pour l'instant utilise -Wfloat-conversion.
  # Les versions récentes de GCC (6.0+) ont l'avertissement de compilation
  # '-Wconversion' qui signale tous les conversions qui peuvent potentiellement
  # poser problèmes (long->int,double->float,...). Les versions plus anciennes
  # (4.9 et 5) ont juste '-Wfloat-conversion' qui signalent uniquement
  # les conversions double->float.
  # Il existe aussi une option option '-Wsign-conversion' qui avertit des
  # conversions de signe mais on ne l'utilise pas pour l'instant car cela fait
  # trop d'avertissements issus des .h.
  check_cxx_compiler_flag("-Wconversion" HAVE_WCONVERSION)
  if(HAVE_WCONVERSION)
    target_compile_options(arccore_build_compile_flags INTERFACE -Wconversion)
  else()
    check_cxx_compiler_flag("-Wfloat-conversion" HAVE_WFLOATCONVERSION)
    if(HAVE_WFLOATCONVERSION)
      target_compile_options(arccore_build_compile_flags INTERFACE -Wfloat-conversion)
    endif()
  endif()

  # Avec certains compilateurs (par exemple clang), l'option '-Wconversion'
  # implique '-Wsign-conversion' ce qui génère pleins d'avertissements inutiles.
  # On supprime donc cette option si on peut.
  check_cxx_compiler_flag("-Wno-sign-conversion" HAVE_NOSIGNCONVERSION)
  if(HAVE_NOSIGNCONVERSION)
    target_compile_options(arccore_build_compile_flags INTERFACE -Wno-sign-conversion)
  endif()

  check_cxx_compiler_flag("-Wpedantic" HAVE_WPEDANTIC)
  if(HAVE_WPEDANTIC)
    # N'utilise pas cet avertissement de compilation si on compile avec CUDA car
    # cela génère pleins d'avertissement sur l'utilisation d'extensions GNU GCC.
    target_compile_options(arccore_build_compile_flags INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>")
  endif()

  check_cxx_compiler_flag("-Wextra" HAVE_WEXTRA)
  if(HAVE_WEXTRA)
    target_compile_options(arccore_build_compile_flags INTERFACE -Wextra)
  endif()

  check_cxx_compiler_flag("-Wshadow" HAVE_WSHADOW)
  if (HAVE_WSHADOW)
    # TODO: remettre cette option mais pour l'instant cela fait trop
    # d'avertissements de compilation.
  endif ()

  # Ajoute de la couleur aux logs de compilations de Ninja.
  check_cxx_compiler_flag("-fdiagnostics-color=always" HAVE_DIAGNOSTICS_COLOR_ALWAYS)
  if(HAVE_DIAGNOSTICS_COLOR_ALWAYS)
    target_compile_options(arccore_build_compile_flags INTERFACE -fdiagnostics-color=always)
  endif()
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
