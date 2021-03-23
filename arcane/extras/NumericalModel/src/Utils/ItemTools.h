// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef ITEMTOOLS_H
#define ITEMTOOLS_H

#include <arcane/IVariable.h>
#include <arcane/Item.h>
#include <arcane/ArcaneVersion.h>
//#include "Utils/ItemVector.h"
#include <arcane/ItemVector.h>

using namespace Arcane;

/*! \brief retourne l'item associé à un localId pour une variable donnée 
 */ 
Item itemFromLocalId(IVariable * ivar, Integer localId);

/*!
  \struct BinaryGroupOperations
  \brief Binary operations between groups
*/
struct BinaryGroupOperations {
  static ItemVector And(ItemGroup a, ItemGroup b);
  static ItemVector Or(ItemGroup a, ItemGroup b);
  static ItemVector Substract(ItemGroup a, ItemGroup b);
  static ItemVector Concatenate(ItemGroup a, ItemGroup b);
};

#endif /* ITEMTOOLS_H */
