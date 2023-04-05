// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellToAllEnvCellConverter.h                                 (C) 2000-2012 */
/*                                                                           */
/* Conversion de 'Cell' en 'AllEnvCell'.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_CELLTOALLENVCELLCONVERTER_H
#define ARCANE_CORE_MATERIALS_CELLTOALLENVCELLCONVERTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IMesh.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"
#include "arcane/core/materials/MatItem.h"
#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/MatItemEnumerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Conversion de 'Cell' en 'AllEnvCell'.

 Les instances de cette classe permettent de convertir une maille \a Cell
 en une maille \a AllEnvCell afin d'avoir les infos sur les matériaux.
 
 La création d'une instance se fait via le gestionnaire de matériaux:
 \code
 * IMeshMaterialMng* mm = ...;
 * CellToAllEnvCellConverter all_env_cell_converter(mm);
 \endcode

 Le coût de la création est faible, équivalent à un appel de fonction
 virtuelle. Il n'est donc pas nul et il est préférable de ne pas construire
 d'instance dans les boucles sur les entités par exemple, mais au dehors.

 Une fois l'instance créée, il est ensuite possible d'utiliser
 l'opérateur [] (operator[]()) pour faire la conversion:
 
 \code
 * CellToAllEnvCellConverter all_env_cell_converter(m_material_mng);
 * Cell cell = ...;
 * ENUMERATE_FACE(iface,allFaces()){
 *   Face face = *iface;
 *   Cell back_cell = face.backCell()
 *   AllEnvCell back_all_env_cell = all_env_cell_converter[back_cell];
 *   info() << "nb env=" << back_all_env_cell.nbEnvironment();
 * }
 \endcode
 
 \warning Les instances de cette classe sont invalidées si la liste des
 mailles matériaux ou milieu change. Dans ce cas, il faut
 refabriquer l'objet:

 \code
 * all_env_cell_converter = CellToAllEnvCellConverter(m_material_mng);
 \endcode
 */
class CellToAllEnvCellConverter
{
 public:

  CellToAllEnvCellConverter(ArrayView<ComponentItemInternal> v)
  : m_all_env_items_internal(v){}

  CellToAllEnvCellConverter(IMeshMaterialMng* mm)
  {
    *this = mm->cellToAllEnvCellConverter();
  }

 public:

  //! Converti une maille \a Cell en maille \a AllEnvCell
  AllEnvCell operator[](Cell c)
  {
    return AllEnvCell(&m_all_env_items_internal[c.localId()]);
  }
  //! Converti une maille \a CellLocalId en maille \a AllEnvCell
  AllEnvCell operator[](CellLocalId c)
  {
    return AllEnvCell(&m_all_env_items_internal[c.localId()]);
  }

 private:

  ArrayView<ComponentItemInternal> m_all_env_items_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Table de connectivité des 'Cell' vers leur(s) 'AllEnvCell' destinée
 *        à une utilisation sur accélérateur.

 Classe singleton qui stoque la connectivité de toutes les mailles 
 \a Cell vers toutes leurs mailles \a AllEnvCell.
 
 On récupère l'instance unique via la méthode statique getInstance() qui 
 la créée le cas échéant (Singleton de Meyers) :
 \code
 * auto& allCell2AllEnvCell(::Arcane::Materials::AllCell2AllEnvCell::getInstance());
 \endcode

 Le coût de l'initialisation est cher, il faut allouer la mémoire et remplir les 
 structures. On parcours toutes les mailles et pour chaque maille on fait 
 appel au CellToAllEnvCellConverter.

 Une fois l'instance créée, elle doit être mise à jour à chaque fois que 
 la topologie des matériaux/environnements change (ce qui est également cher).
 
 \code
 * auto& ac2aec(::Arcane::Materials::AllCell2AllEnvCell::getInstance());
 * command << RUNCOMMAND_ENUMERATE(Cell, cid, allCells()) {
 *   for (Int32 i(0); i < ac2aec->getAllCell2AllEnvCellTable()[cid].size(); ++i) {
 *     const auto& mvi(ac2aec->getAllCell2AllEnvCellTable()[cid][i]);
 *     sum2 += in_menv_var2[mvi]/in_menv_var2_g[cid];
 *     ...
 *   }
 * };
 \endcode
 
 */
class AllCell2AllEnvCell
{
 private:

  AllCell2AllEnvCell()
  : m_mm(nullptr), m_alloc(nullptr), m_nb_allcell(0), m_allcell_allenvcell(nullptr){}

  void reset()
  {
    if (m_allcell_allenvcell) {
      for (Int64 i(0); i < m_nb_allcell; ++i) {
        if (!m_allcell_allenvcell[i].empty()) {
          m_alloc->deallocate(m_allcell_allenvcell[i].data());
        }
      }
      m_alloc->deallocate(m_allcell_allenvcell);
      m_allcell_allenvcell = nullptr;
    }
    m_mm = nullptr;
    m_alloc = nullptr;
    m_nb_allcell = 0;
  }

  ~AllCell2AllEnvCell()
  {
    reset();
  }

  
 public:
  //! Copies interdites
  AllCell2AllEnvCell(const AllCell2AllEnvCell&) = delete;
  AllCell2AllEnvCell& operator=(const AllCell2AllEnvCell&) = delete;

  //! Méthode d'accès au singleton (créé l'objet si nécessaire)
  // (singleton à la Meyers)
  static AllCell2AllEnvCell& getInstance()
  {
    static AllCell2AllEnvCell instance;
    return instance;
  }

  /*!
   * La fonction d'initialisation doit attendre que les informations relatives
   * aux matériaux soient finalisées
   */
  void init(IMeshMaterialMng* mm, IMemoryAllocator* alloc)
  {
    m_mm = mm;
    m_alloc = alloc;
    m_nb_allcell = m_mm->mesh()->allCells().size();
    
    CellToAllEnvCellConverter all_env_cell_converter(m_mm);

    m_allcell_allenvcell = reinterpret_cast<Span<ComponentItemLocalId>*>(m_alloc->allocate(sizeof(Span<ComponentItemLocalId>) * m_nb_allcell));

    ENUMERATE_CELL(icell, m_mm->mesh()->allCells())
    {
      Cell cell = *icell;
      Integer cid = cell.localId();
      AllEnvCell all_env_cell = all_env_cell_converter[cell];
      ComponentItemLocalId* env_cells(nullptr);
      Span<ComponentItemLocalId> env_cells_span;
      Integer nb_env(all_env_cell.nbEnvironment());
      if (nb_env) {
        env_cells = reinterpret_cast<ComponentItemLocalId*>(m_alloc->allocate(sizeof(ComponentItemLocalId) * nb_env));
        Integer i(0);
        ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell) {
          EnvCell ev = *ienvcell;
          env_cells[i] = ComponentItemLocalId(ev._varIndex());
          ++i;
        }
        env_cells_span = Span<ComponentItemLocalId>(env_cells, nb_env);
      }
      m_allcell_allenvcell[cid] = env_cells_span;
    }
  }

  // Rien d'intelligent ici, on refait tout. Il faut voir dans un 2nd temps 
  // pour faire qqch de plus malin et donc certainement plus rapide...
  // SI on garde cette classe et ce concept... ça m'étonnerait...
  void bruteForceUpdate(Int32ConstArrayView ids)
  {
    // A priori, je ne pense pas que le nb de maille ait changé quand on fait un 
    // ForceRecompute. Mais ça doit arriver ailleurs... le endUpdate ?
    if (m_nb_allcell != m_mm->mesh()->allCells().size()) {

      // TODO: Je met un fatal, à supprimer une fois bien testé/exploré
      ARCANE_FATAL("The number of cells has changed since initialization of AllCell2AllEnvCell.");

      IMeshMaterialMng* mm(m_mm);
      IMemoryAllocator* alloc(m_alloc);
      reset();
      getInstance().init(mm, alloc);
    } else {
      // Si le nb de maille n'a pas changé, on reconstruit en fonction de la liste de maille
      CellToAllEnvCellConverter all_env_cell_converter(m_mm);
      for (Integer i(0), n=ids.size(); i<n; ++i) {
        CellLocalId lid = static_cast<CellLocalId>(ids[i]);
        // Si c'est pas vide, on efface et on refait
        if (!m_allcell_allenvcell[lid].empty()) {
          m_alloc->deallocate(m_allcell_allenvcell[lid].data());
        }
        AllEnvCell all_env_cell = all_env_cell_converter[lid];
        ComponentItemLocalId* env_cells(nullptr);
        Span<ComponentItemLocalId> env_cells_span;
        Integer nb_env(all_env_cell.nbEnvironment());
        if (nb_env) {
          env_cells = reinterpret_cast<ComponentItemLocalId*>(m_alloc->allocate(sizeof(ComponentItemLocalId) * nb_env));
          Integer i(0);
          ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell) {
            EnvCell ev = *ienvcell;
            env_cells[i] = ComponentItemLocalId(ev._varIndex());
            ++i;
          }
          env_cells_span = Span<ComponentItemLocalId>(env_cells, nb_env);
        }
        m_allcell_allenvcell[lid] = env_cells_span;
      }
    }
  }

  /*!
   * Méthode d'accès à la table de "connectivité" cell -> all env cells
   */
  ARCCORE_HOST_DEVICE const Span<ComponentItemLocalId>* getAllCell2AllEnvCellTable() const
  {
    return m_allcell_allenvcell;
  }

 private:
  IMeshMaterialMng* m_mm;
  IMemoryAllocator* m_alloc;
  Integer m_nb_allcell;
  Span<ComponentItemLocalId>* m_allcell_allenvcell;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
class Cell2AllComponentCellEnumerator
{
  friend class EnumeratorTracer;

 public:
  using index_type = Span<ComponentItemLocalId>::index_type;
  using size_type = Span<ComponentItemLocalId>::size_type;

 public:
  // La version CPU permet de vérifier qu'on a bien fait l'init avant l'ENUMERATE
  ARCCORE_HOST_DEVICE Cell2AllComponentCellEnumerator(Integer cell_id)
  : m_cid(cell_id), m_index(0)
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  , m_ptr(&::Arcane::Materials::AllCell2AllEnvCell::getInstance().getAllCell2AllEnvCellTable()[cell_id])
  , m_size((*m_ptr).size())
  {
  }
#else
  , m_ptr(nullptr), m_size(0)
  {
    auto& allCell2AllEnvCell(::Arcane::Materials::AllCell2AllEnvCell::getInstance());
    if (allCell2AllEnvCell.getAllCell2AllEnvCellTable()) {
      m_ptr = &::Arcane::Materials::AllCell2AllEnvCell::getInstance().getAllCell2AllEnvCellTable()[cell_id];
      m_size = (*m_ptr).size();
    } else {
      ARCANE_FATAL("Must initialize AllCell2AllEnvCell singleton before using ENUMERATE_ALLENVCELL");
    }
  }
#endif

  ARCCORE_HOST_DEVICE inline void operator++() { ++m_index; }

  ARCCORE_HOST_DEVICE inline bool hasNext() const { return m_index<m_size; }

  ARCCORE_HOST_DEVICE inline ComponentItemLocalId& operator*() const { return (*m_ptr)[m_index]; }

 private:
  Integer m_cid;
  index_type m_index;
  const Span<ComponentItemLocalId>* m_ptr;
  size_type m_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: Très certainement à déplacer ailleurs si on garde ce proto
#define A_ENUMERATE_CELL_ALLCOMPONENTCELL(_EnumeratorClassName,iname,cell_id) \
  for( A_TRACE_COMPONENT(_EnumeratorClassName) iname((::Arcane::Materials::_EnumeratorClassName)(cell_id) A_TRACE_ENUMERATOR_WHERE); iname.hasNext(); ++iname )

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro pour itérer sur tous les milieux d'une maille à l'intérieur.
 *        Version "brute et légère" ENUMERATE_CELL_ENVCELL, destinée à un 
 *        emploi sur accélérateur, i.e. au sein d'un RUN_COMMAND...
 *
 * \param iname nom de la variable (type MatVarIndex) permettant l'accès aux 
 *              données.
 * \param cid identifiant de la maille (type Integer).
 */
// TODO: Très certainement à déplacer ailleurs si on garde ce proto
#define ENUMERATE_CELL_ALLENVCELL(iname,cid) \
  A_ENUMERATE_CELL_ALLCOMPONENTCELL(Cell2AllComponentCellEnumerator,iname,cid)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

