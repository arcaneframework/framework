//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;

namespace Arcane.Curves
{
  /// <summary>
  /// Indique comment traiter les courbes multi-valeurs
  /// </summary>
  public enum MultiCurveBehaviour
  {
    // Ignore les courbes avec plusieurs valeurs
    Ignore,
    // Traite les courbes comme N courbes classiques
    ReadAsSeveralMonoValue
  }


  /// <summary>
  /// Paramêtres pour modifier le comportement de lecture des courbes.
  /// </summary>
  public class CaseReaderSettings
  {
    /// <summary>
    /// Indique comment gérer les courbes multi-valeurs. Par défaut, vaut MultiCurveBehaviour.Ignore
    /// </summary>
    public MultiCurveBehaviour MultiCurveBehavior;

    public CaseReaderSettings ()
    {
      MultiCurveBehavior = MultiCurveBehaviour.Ignore;
    }
  }
}

