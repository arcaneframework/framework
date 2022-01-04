// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/tests/CartesianMeshV2TestUtils.h"

#include "arcane/cartesianmesh/v2/CartesianTypes.h"
#include "arcane/cartesianmesh/v2/CartesianGrid.h"
#include "arcane/cartesianmesh/v2/CartesianNumbering.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::CartesianMesh::V2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
template <typename IdType>
void test_CartesianNumbering(const IdType (&nitem)[3], Integer dimension,
                             IdType first_item_id)
{
  CartesianNumbering<IdType> cart_numb;

  cart_numb.initNumbering(nitem, dimension, first_item_id);

  ASSERT_EQ(cart_numb.dimension(), dimension) << "Bad dimension";

  IdType nitem_comp = 1;

  for (Integer dir = 0; dir < dimension; ++dir) {
    ASSERT_EQ(cart_numb.nbItem3()[dir], nitem[dir]) << "(A) Incorrect nb of items for the direction";
    ASSERT_EQ(cart_numb.nbItemDir(dir), nitem[dir]) << "(B) Incorrect nb of items for the direction";

    nitem_comp *= cart_numb.nbItemDir(dir);
  }
  for (Integer dir = dimension; dir < 3; ++dir) {
    ASSERT_EQ(cart_numb.nbItem3()[dir], 1) << "(C) Incorrect nb of items for the direction";
    ASSERT_EQ(cart_numb.nbItemDir(dir), 1) << "(D) Incorrect nb of items for the direction";
  }
  ASSERT_EQ(cart_numb.nbItem(), nitem_comp) << "Incorrect total nb of items";

  // First item
  ASSERT_EQ(cart_numb.firstId(), first_item_id) << "(A) Incorrect first item id";
  ASSERT_EQ(cart_numb.id(0, 0, 0), first_item_id) << "(B) Incorrect first item id";
  IdType idx0[3] = { 0, 0, 0 };
  ASSERT_EQ(cart_numb.id(idx0), first_item_id) << "(C) Incorrect first item id";
  ASSERT_EQ(cart_numb.id(IdxType{ 0, 0, 0 }), first_item_id) << "(D) Incorrect first item id";

  IdType idx00[3] = { -1, -1, -1 };
  cart_numb.ijk(first_item_id, idx00);
  ASSERT_EQ(idx00[0], 0) << "(A) Incorrect I index";
  ASSERT_EQ(idx00[1], 0) << "(A) Incorrect J index";
  ASSERT_EQ(idx00[2], 0) << "(A) Incorrect K index";

  IdxType idx000 = cart_numb.ijk(first_item_id);
  ASSERT_EQ(idx000[0], 0) << "(B) Incorrect I index";
  ASSERT_EQ(idx000[1], 0) << "(B) Incorrect J index";
  ASSERT_EQ(idx000[2], 0) << "(B) Incorrect K index";

  ASSERT_EQ(cart_numb.idxDir0(first_item_id), 0) << "(C) Incorrect first I index";
  ASSERT_EQ(cart_numb.idxDir1(first_item_id), 0) << "(C) Incorrect first J index";
  ASSERT_EQ(cart_numb.idxDir2(first_item_id), 0) << "(C) Incorrect first K index";

  // Last item
  IdType last_item_id = first_item_id + cart_numb.nbItem() - 1;
  IdType litem[3] = {
    cart_numb.nbItemDir(0) - 1,
    cart_numb.nbItemDir(1) - 1,
    cart_numb.nbItemDir(2) - 1
  };
  ASSERT_EQ(cart_numb.id(litem[0], litem[1], litem[2]), last_item_id) << "(A) Incorrect last item id";
  ASSERT_EQ(cart_numb.id(litem), last_item_id) << "(B) Incorrect last item id";
  ASSERT_EQ(cart_numb.id(IdxType{ litem[0], litem[1], litem[2] }), last_item_id) << "(C) Incorrect last item id";

  IdType idx1[3] = { -1, -1, -1 };
  cart_numb.ijk(last_item_id, idx1);
  ASSERT_EQ(idx1[0], cart_numb.nbItemDir(0) - 1) << "(A) Incorrect last I index";
  ASSERT_EQ(idx1[1], cart_numb.nbItemDir(1) - 1) << "(A) Incorrect last J index";
  ASSERT_EQ(idx1[2], cart_numb.nbItemDir(2) - 1) << "(A) Incorrect last K index";

  // Check numbering
  IdType cur_id = cart_numb.firstId();
  for (IdType k(0); k < cart_numb.nbItemDir(2); ++k) {
    for (IdType j(0); j < cart_numb.nbItemDir(1); ++j) {
      for (IdType i(0); i < cart_numb.nbItemDir(0); ++i) {

        IdType ida = cart_numb.id(i, j, k);
        ASSERT_EQ(ida, cur_id) << "(A) Incorrect current id";

        IdType idxb[3] = { i, j, k };
        IdType idb = cart_numb.id(idxb);
        ASSERT_EQ(idb, cur_id) << "(B) Incorrect current id";

        IdType idc = cart_numb.id(IdxType{ i, j, k });
        ASSERT_EQ(idc, cur_id) << "(C) Incorrect current id";

        // Inverse operation
        IdType idd[3] = { -1, -1, -1 };
        cart_numb.ijk(cur_id, idd);
        ASSERT_EQ(idd[0], i) << "(A) Incorrect current I index";
        ASSERT_EQ(idd[1], j) << "(A) Incorrect current J index";
        ASSERT_EQ(idd[2], k) << "(A) Incorrect current K index";

        IdxType ide = cart_numb.ijk(cur_id);
        ASSERT_EQ(ide[0], static_cast<Int64>(i)) << "(B) Incorrect current I index";
        ASSERT_EQ(ide[1], static_cast<Int64>(j)) << "(B) Incorrect current J index";
        ASSERT_EQ(ide[2], static_cast<Int64>(k)) << "(B) Incorrect current K index";

        cur_id++;
      }
    }
  }

  // Shift to neighbors
  for (Integer dir = 0; dir < dimension; ++dir) {
    ASSERT_EQ(cart_numb.delta3()[dir], cart_numb.deltaDir(dir)) << "Shifting numbers not equal";

    IdType nitemD[3] = { cart_numb.nbItemDir(0), cart_numb.nbItemDir(1), cart_numb.nbItemDir(2) };
    nitemD[dir] -= 1;
    IdType delta = cart_numb.deltaDir(dir);

    for (IdType k(0); k < nitemD[2]; ++k) {
      for (IdType j(0); j < nitemD[1]; ++j) {
        for (IdType i(0); i < nitemD[0]; ++i) {

          IdType id_nei_comp = cart_numb.id({ i, j, k }) + delta; // neighbor by shifting

          IdxType idxN{ i, j, k };
          idxN[dir] += 1;
          IdType id_nei = cart_numb.id(idxN);
          ASSERT_EQ(id_nei, id_nei_comp) << "Incorrect neighbor id";
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(CartesianMeshV2, TestCartesianNumbering)
{
  std::cout << "TEST_CARTESIANMESHV2 LocalIdType for dimension=2\n";
  {
    LocalIdType3 nitem = { 5, 4, 0 };
    test_CartesianNumbering<LocalIdType>(nitem, /*dim=*/2, /*first_id=*/42);
  }
  std::cout << "TEST_CARTESIANMESHV2 LocalIdType for dimension=3\n";
  {
    LocalIdType3 nitem = { 5, 4, 3 };
    test_CartesianNumbering<LocalIdType>(nitem, /*dim=*/3, /*first_id=*/100);
  }
  std::cout << "TEST_CARTESIANMESHV2 UniqueIdType for dimension=2\n";
  {
    UniqueIdType3 nitem = { 5, 4, 0 };
    UniqueIdType first_id = (UniqueIdType{ 1 } << 33) + UniqueIdType{ 42 };
    test_CartesianNumbering<UniqueIdType>(nitem, /*dim=*/2, first_id);
  }
  std::cout << "TEST_CARTESIANMESHV2 UniqueIdType for dimension=3\n";
  {
    UniqueIdType3 nitem = { 5, 4, 3 };
    UniqueIdType first_id = (UniqueIdType{ 1 } << 33) + UniqueIdType{ 100 };
    test_CartesianNumbering<UniqueIdType>(nitem, /*dim=*/3, first_id);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
template <typename IdType>
void test_CartesianGrid(const IdType (&ncell)[3], Integer dimension,
                        bool only_center)
{
  CartesianGrid<IdType> cart_grid(ncell, dimension);

  ASSERT_EQ(cart_grid.dimension(), dimension) << "Incorrect dimension";

  ASSERT_EQ(&cart_grid.cartNumCell(), cart_grid.cartNumCellPtr()) << "(A) Pointers not equal";
  ASSERT_EQ(&cart_grid.cartNumNode(), cart_grid.cartNumNodePtr()) << "(B) Pointers not equal";
  ASSERT_EQ(&cart_grid.cartNumFace3(), cart_grid.cartNumFace3Ptr()) << "(C) Pointers not equal";
  for (Integer dir = 0; dir < dimension; ++dir) {
    ASSERT_EQ(&cart_grid.cartNumFace(dir), &(cart_grid.cartNumFace3()[dir])) << "(D) Pointers not equal";
  }

  auto cell_numb = cart_grid.cartNumCell();
  auto node_numb = cart_grid.cartNumNode();

  for (Integer dir = 0; dir < dimension; ++dir) {
    auto face_numb_d = cart_grid.cartNumFace3()[dir];

    ASSERT_EQ(cell_numb.nbItemDir(dir), ncell[dir]) << "Incorrect nb of cells";
    ASSERT_EQ(cell_numb.nbItemDir(dir) + 1, node_numb.nbItemDir(dir)) << "Incorrect nb of cells/nodes";
    ASSERT_EQ(cell_numb.nbItemDir(dir) + 1, face_numb_d.nbItemDir(dir)) << "Incorrect nb of cells/faces dir";

    for (Integer idir = 1; idir < dimension; ++idir) {
      Integer dtrans = (dir + idir) % dimension;
      ASSERT_EQ(cell_numb.nbItemDir(dtrans), face_numb_d.nbItemDir(dtrans)) << "Incorrect nb of cells/faces dtrans";
    }
  }

  // Connectivity
  auto lbd_c2n = [&](IdType i, IdType j, IdType k) {
    auto cid = cell_numb.id({ i, j, k });
    auto nid = node_numb.id({ i, j, k });

    IdType delta_c2n = node_numb.firstId() - cell_numb.firstId() +
    j + k * (cell_numb.nbItemDir(0) + cell_numb.nbItemDir(1) + 1);
    ASSERT_EQ(cid + delta_c2n, nid) << "Incorrect Cell/Node ids";
  };
  auto lbd_c2f = [&](Integer dir,
                     const CartesianNumbering<IdType>& face_numb_d,
                     IdType i, IdType j, IdType k) {
    auto cid = cell_numb.id({ i, j, k });
    auto fid = face_numb_d.id({ i, j, k });

    IdType delta = face_numb_d.firstId() - cell_numb.firstId() +
    (dir == 0 ? j + k * cell_numb.nbItemDir(1) : (dir == 1 ? k * cell_numb.nbItemDir(0) : 0));
    ASSERT_EQ(cid + delta, fid) << "Incorrect Cell/Face ids";
  };

  if (only_center) {
    IdType i = cell_numb.nbItemDir(0) / 2;
    IdType j = cell_numb.nbItemDir(1) / 2;
    IdType k = cell_numb.nbItemDir(2) / 2;
    // Cell => Node
    lbd_c2n(i, j, k);
    // Cell => Face
    for (Integer dir = 0; dir < dimension; ++dir) {
      lbd_c2f(dir, cart_grid.cartNumFace3()[dir], i, j, k);
    }
  }
  else {
    // Cell => Node
    for (IdType k(0); k < cell_numb.nbItemDir(2); ++k) {
      for (IdType j(0); j < cell_numb.nbItemDir(1); ++j) {
        for (IdType i(0); i < cell_numb.nbItemDir(0); ++i) {
          lbd_c2n(i, j, k);
        }
      }
    }
    // Cell => Face
    for (Integer dir = 0; dir < dimension; ++dir) {
      auto face_numb_d = cart_grid.cartNumFace3()[dir];

      for (IdType k(0); k < cell_numb.nbItemDir(2); ++k) {
        for (IdType j(0); j < cell_numb.nbItemDir(1); ++j) {
          for (IdType i(0); i < cell_numb.nbItemDir(0); ++i) {
            lbd_c2f(dir, face_numb_d, i, j, k);
          }
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(CartesianMeshV2, TestCartesianGrid)
{
  std::cout << "TEST_CARTESIANMESHV2_CARTESIANGRID LocalIdType for dimension=2\n";
  {
    LocalIdType3 ncell = { 5, 4, 0 };
    test_CartesianGrid<LocalIdType>(ncell, /*dim=*/2, /*only_center=*/false);
  }
  std::cout << "TEST_CARTESIANMESHV2_CARTESIANGRID LocalIdType for dimension=3\n";
  {
    LocalIdType3 ncell = { 5, 4, 3 };
    test_CartesianGrid<LocalIdType>(ncell, /*dim=*/3, /*only_center=*/false);
  }
  std::cout << "TEST_CARTESIANMESHV2_CARTESIANGRID UniqueIdType for dimension=2\n";
  {
    UniqueIdType3 ncell = { 150000, 75000, 0 };
    test_CartesianGrid<UniqueIdType>(ncell, /*dim=*/2, /*only_center=*/true);
  }
  std::cout << "TEST_CARTESIANMESHV2_CARTESIANGRID UniqueIdType for dimension=3\n";
  {
    UniqueIdType3 ncell = { 3000, 2000, 1500 };
    test_CartesianGrid<UniqueIdType>(ncell, /*dim=*/3, /*only_center=*/true);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Effecttue des instantiations explicites pour tester la compilation.
template class Arcane::CartesianMesh::V2::CartesianGrid<Arcane::Int32>;
template class Arcane::CartesianMesh::V2::CartesianGrid<Arcane::Int64>;

template class Arcane::CartesianMesh::V2::CartesianNumbering<Arcane::Int32>;
template class Arcane::CartesianMesh::V2::CartesianNumbering<Arcane::Int64>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
