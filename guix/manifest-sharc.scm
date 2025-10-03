(use-modules (guix build-system gnu)
             (ice-9 match))

(define stdenv
  (map (lambda* (pkg)
         (match pkg
           ((_ value _ ...)
            value)))
       (standard-packages)))

(concatenate-manifests (list (specifications->manifest (list "bash"
                                                        "git"
                                                        "cmake"
                                                        "pkg-config"
                                                        "googletest"
                                                        "openmpi@4.1.6"
                                                        "glib"
                                                        "libxml2"
                                                        "dotnet@8"
							"mono@6.12.0.206"
                                                        "openblas"
                                                        "boost"
                                                        "petsc-openmpi@3.21.4"
							"hypre-openmpi@2.32.0"
                                                        "eigen"
							"hdf5-parallel-openmpi@1.14.6"
                                                        "gfortran-toolchain"))
                             (packages->manifest stdenv)))
