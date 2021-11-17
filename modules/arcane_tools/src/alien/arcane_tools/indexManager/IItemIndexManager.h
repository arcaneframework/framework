#ifndef ALIEN_IITEMINDEX_MANAGER_H
#define ALIEN_IITEMINDEX_MANAGER_H

#include "alien/AlienArcaneToolsPrecomp.h"
#include "alien/arcane_tools/IIndexManager.h"

namespace Arcane {
class IItemFamily;
class ItemGroup;
class Item;
class ItemVectorView;
} // namespace Arcane

/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTools {

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * Voir \ref secLinearSolver "Description des services de solveurs
   * linéaires" pour plus de détails
   *
   * \todo Pour gérer les inconnues vectoriel, mettre un argument additionnel qui
   * est le sous-indice (vectoriel) avec une valeur par défaut de 0 (ou
   * bien un proto différent)
   */
  class IItemIndexManager : public IIndexManager
  {
   public:
    typedef IIndexManager BaseType;

   public:
    //! Constructeur par défaut
    IItemIndexManager()
    : BaseType()
    {
    }

    //! Destructeur
    virtual ~IItemIndexManager() {}

   public:
    using IIndexManager::buildScalarIndexSet;

    //! Construit une nouvelle entrée scalaire sur des items du maillage
    virtual ScalarIndexSet buildScalarIndexSet(
        const Arccore::String name, Arcane::IItemFamily* item_family) = 0;
    virtual void defineIndex(ScalarIndexSet& set, const Arcane::ItemGroup& itemGroup) = 0;
    virtual ScalarIndexSet buildScalarIndexSet(
        const Arccore::String name, const Arcane::ItemGroup& itemGroup) = 0;

    //! Construit une nouvelle entrée vectorielle sur des items du maillage
    /*! L'implémentation actuelle considére le multi-scalaire comme du vectoriel */
    virtual VectorIndexSet buildVectorIndexSet(const Arccore::String name,
        const Arcane::ItemGroup& itemGroup, const Arccore::Integer n) = 0;

    //! Demande de dé-indexation d'une partie d'une entrée
    /*! Utilisable uniquement avant prepare */
    virtual void removeIndex(
        const ScalarIndexSet& entry, const Arcane::ItemGroup& itemGroup) = 0;

    virtual Arccore::Integer getIndex(
        const Entry& entry, const Arcane::Item& item) const = 0;

    //! Consultation vectorielle d'indexation d'une entrée (après prepare)
    virtual void getIndex(const ScalarIndexSet& entry,
        const Arcane::ItemVectorView& items,
        Arccore::ArrayView<Arccore::Integer> indexes) const = 0;
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ALIEN_IITEMINDEX_MANAGER_H */
