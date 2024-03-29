﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef ARCGEOSIM_DISCRETEOPERATOR_DIVKGRADDISCRETEOPERATORIMPL_DISCRETEOPERATORPROPERTY_H
#define ARCGEOSIM_DISCRETEOPERATOR_DIVKGRADDISCRETEOPERATORIMPL_DISCRETEOPERATORPROPERTY_H
/* Author : dipietrd at Wed Sep 10 15:59:59 2008
 * Generated by createNew
 */

struct DiscreteOperatorProperty {
  enum eOption {
    O_NONE                           = 0,
    O_DISABLE_STENCIL_COMPUTATION    = (1 << 0), //! Disable stencil computation
    O_DISABLE_CELL_GROUP_COMPUTATION = (1 << 1), //! Disable cell group computation
    O_DISABLE_FACE_GROUP_COMPUTATION = (1 << 2), //! Disable face group computation
    O_FACES_ITEMGROUP                = (1 << 3), //! Store faces as an item group
    O_FACES_ITEMVECTOR               = (1 << 4)  //! Store faces as an item vector
  };

  enum eProperty {
    P_NONE              = 0,
    P_CELLS             = (1 << 0), //! The group of all cells
    P_FACES             = (1 << 1)  //! The group of all (internal and boundary) faces
  };

  enum eStatus {
    S_NONE              = 0,
    S_INITIALIZED       = (1 << 0), //! Service initialized
    S_PREPARED          = (1 << 1), //! Service prepared
    S_FORMED            = (1 << 2), //! Operator has been formed at least once after last prepare
    S_FINALIZED         = (1 << 3)  //! Service finalized
  };
};

#endif /* ARCGEOSIM_DISCRETEOPERATOR_DIVKGRADDISCRETEOPERATORIMPL_DISCRETEOPERATORPROPERTY_H */
