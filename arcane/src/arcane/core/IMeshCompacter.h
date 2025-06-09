// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshCompacter.h                                            (C) 2000-2025 */
/*                                                                           */
/* Gestion d'un compactage de familles du maillage.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHCOMPACTER_H
#define ARCANE_CORE_IMESHCOMPACTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion d'un compactage de familles du maillage.
 *
 * Les instances de cette classe sont créée via le gestionnaire
 * IMeshCompactMng. Un seul compactage peut avoir lieu à la fois.
 *
 * Par compactage, on entend toute modification de la numérotation locale
 * des entités d'une famille. Il peut donc rester des trous dans la numérotation
 * après appel à un compactage (même si actuellement ce n'est pas le cas
 * des implémentations disponibles dans %Arcane).
 *
 * Le compactage concerne soit toutes les familles d'un maillage, soit
 * une seule famille. La méthode families() permet de retourner la
 * liste des familles compactées.
 *
 * Même si une famille n'est pas compactée directement, elle participe à
 * certaines opérations du compactage car elle peut faire référence à des
 * entités compactées.
 *
 * Les différentes opérations d'un compactage sont les suivantes:
 * 1. beginCompact():calcul de la nouvelle numération locale des entités
 * des familles compactées. Après appel à cette méthode, il est possible
 * d'appeler findCompactInfos() pour obtenir pour une famille les
 * correspondances entre nouveaux et anciens numéros locaux.
 * 2. compactVariablesAndGroups(): mise à jour des groupes et des variables
 * des familles compactées en fonction de cette nouvelle numérotation.
 * 3. updateInternalReferences(): mise à jour des références aux entités.
 * Cela concerne toutes les familles et pas seulement celles compactées.
 * 4. endCompact(): finalise le compactage des familles. Après appel à cette
 * méthode il n'est plus possible de récupérer les informations de correspondance
 * via findCompactInfos().
 * 5. finalizeCompact(): notification à toutes les familles que le compactage
 * est terminé. Cela permet par exemple de faire un nettoyage ou de mettre
 * à jour certaines informations.
 *
 * La méthode doAllActions() permet de faire toutes ces phases en une seule fois.
 * C'est la méthode recommandé pour effectuer un compactage. Le code suivant
 * montre comment effectuer un compactage sur toutes les familles:
 *
 * \code
 *
 * IMeshCompactMng* compact_mng = mesh()->_compactMng();
 * IMeshCompacter* compacter = compact_mng->beginCompact();
 *
 * try{
 *   compacter->doAllActions();
 * }
 * catch(...){
 *   compact_mng->endCompact();
 *   throw;
 * }
 * compact_mng->endCompact();
 *
 * \endcode
 */
class ARCANE_CORE_EXPORT IMeshCompacter
{
 public:

  //! Indique les différentes phases du compactage
  enum class ePhase
  {
    Init = 0,
    BeginCompact,
    CompactVariableAndGroups,
    UpdateInternalReferences,
    EndCompact,
    Finalize,
    Ended
  };

 public:

  virtual ~IMeshCompacter() = default; //!< Libère les ressources

 public:

  //! Exécute successivement toutes les actions de compactage.
  virtual void doAllActions() = 0;

  virtual void beginCompact() = 0;
  virtual void compactVariablesAndGroups() = 0;
  virtual void updateInternalReferences() = 0;
  virtual void endCompact() = 0;
  virtual void finalizeCompact() = 0;

  //! Maillage associé à ce compacter.
  virtual IMesh* mesh() const = 0;

  /*!
   * \brief Informations de compactage pour la famille \a family.
   *
   * Le pointeur retourné peut être nul si la famille spécifiée ne fait
   * pas partie des familles compactées.
   */
  virtual const ItemFamilyCompactInfos* findCompactInfos(IItemFamily* family) const = 0;

  //! Phase de l'échange dans laquelle on se trouve.
  virtual ePhase phase() const = 0;

  /*!
   * \brief Indique s'il faut trier les entités lors du compactage.
   * \pre phase()==ePhase::Init.
   */
  virtual void setSorted(bool v) = 0;

  //! Indique si souhaite trier les entités en plus de les compacter.
  virtual bool isSorted() const = 0;

  //! Familles dont les entités sont compactées.
  virtual ItemFamilyCollection families() const = 0;

  //! \internal
  virtual void _setCompactVariablesAndGroups(bool v) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
