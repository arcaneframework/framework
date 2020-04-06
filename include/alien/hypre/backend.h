#pragma once

#include <alien/core/backend/BackEnd.h>
#include <alien/core/backend/LinearSolver.h>
#include <alien/core/backend/LinearAlgebra.h>

namespace Alien::Hypre {
    class Matrix;

    class Vector;

    class Options;

    extern IInternalLinearSolver<Matrix, Vector> *InternalLinearSolverFactory(const Options &options);

    extern IInternalLinearSolver<Matrix, Vector> *InternalLinearSolverFactory();

    extern IInternalLinearAlgebra<Matrix, Vector> *InternalLinearAlgebraFactory();
}

namespace Alien {
    namespace BackEnd {
        namespace tag {
            struct hypre {
            };
        }
    }

    template<>
    struct AlgebraTraits<BackEnd::tag::hypre> {
        // types
        using matrix_type = Hypre::Matrix;
        using vector_type = Hypre::Vector;
        using options_type = Hypre::Options;
        using algebra_type = IInternalLinearAlgebra<matrix_type, vector_type>;
        using solver_type = IInternalLinearSolver<matrix_type, vector_type>;

        // factory to build algebra
        static auto *algebra_factory() {
            return Hypre::InternalLinearAlgebraFactory();
        }

        // factories to build solver
        static auto *solver_factory(const options_type &options) {
            return Hypre::InternalLinearSolverFactory(options);
        }

        // factories to build default solver
        static auto *solver_factory() {
            return Hypre::InternalLinearSolverFactory();
        }

        static BackEndId name() { return "hypre"; }
    };

}

// user interface
namespace Alien::Hypre {
    using LinearSolver = Alien::LinearSolver<BackEnd::tag::hypre>;
    using LinearAlgebra = Alien::LinearAlgebra<BackEnd::tag::hypre>;
}
