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
                                                        "openmpi"
                                                        "glib"
                                                        "libxml2"
                                                        "dotnet@8"
                                                        "openblas"
                                                        "boost"
                                                        "gfortran-toolchain"))
                             (packages->manifest stdenv)))
