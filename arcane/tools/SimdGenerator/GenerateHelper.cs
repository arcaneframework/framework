//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;

namespace SimdGenerator
{
  public static class GenerateHelper
  {
    public static BinaryOperation[] BinaryOperations { get; private set; }

    public static UnaryOperation[] UnaryOperations { get; private set; }

    static GenerateHelper ()
    {
      BinaryOperations = new BinaryOperation[] {
        BinaryOperation.Sub,
        BinaryOperation.Add,
        BinaryOperation.Mul,
        BinaryOperation.Div,
        BinaryOperation.Min,
        BinaryOperation.Max
      };

      UnaryOperations = new UnaryOperation[] {
        UnaryOperation.SquareRoot,
        UnaryOperation.Exponential,
        UnaryOperation.Log10,
        UnaryOperation.UnaryMinus
      };

    }
  }
}

