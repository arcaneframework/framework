#ifndef BOUNDARYCONDITIONTYPES_H
#define BOUNDARYCONDITIONTYPES_H

/*!
  \struct BoundaryConditionTypes
  \author Daniele Di Pietro <daniele-antonio.di-pietro@ifp.fr>
  \date 2007-30-7
  \brief Boundary condition types
*/

struct BoundaryConditionTypes {
  enum eType {
    Inflow,    //! Inflow boundary condition
    Dirichlet, //! Dirichlet boundary condition
    Neumann,   //! Neumann boundary condition
    Robin      //! Robin boundary condition
  };
};

#endif
