// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SequentialSection.h                                         (C) 2000-2025 */
/*                                                                           */
/* Section du code à exécuter séquentiellement.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SEQUENTIALSECTION_H
#define ARCANE_CORE_SEQUENTIALSECTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ParallelFatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Section de code à exécuter séquentiellement.
 *
 * Une instance de cette classe permet à une partie du code de s'exécuter
 * un peu comme si le code était séquentiel. Le code compris dans la durée
 * de vie de cet objet est d'abord exécuté sur le proc 0, puis si tout est
 * ok sur les autres. Cela permet lorsque le code exécuter est le même
 * partout (par exemple la lecture du jeu de données) de le vérifier une
 * fois et en cas d'erreur d'afficher une seule fois les messages.
 *
 * Comme les erreurs éventuelles ne sont affichées que par un seul
 * sous-domaine, cette classe ne doit être utilisée que lorsqu'on est
 * certain que tout les sous-domaines font le même traitement faute
 * de quoi les erreurs ne seront pas reconnues.
 *
 * De plus, comme tous les sous-domaines bloquent tant que le premier
 * sous-domaine n'a pas fini d'exécuter le code, il faut ne faut pas
 * faire d'appel au gestionnaire de parallélisme dans cette partie.
 *
 * En cas d'erreur, une exception de type ExParallelFatalError est
 * envoyée dans le destructeur.
 *
 \code
 * {
 *   SequentialSection ss(pm);
 *   ... // Code exécuté séquentiellement.
 *   ss.setError(true);
 * }
 \endcode
 *
 */
class ARCANE_CORE_EXPORT SequentialSection
{
 public:

  explicit SequentialSection(IParallelMng*);
  explicit SequentialSection(ISubDomain*);
  ~SequentialSection() ARCANE_NOEXCEPT_FALSE;

 public:

  void setError(bool is_error);

 private:

  IParallelMng* m_parallel_mng = nullptr;
  bool m_has_error = false;

  void _init();
  void _sendError();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

