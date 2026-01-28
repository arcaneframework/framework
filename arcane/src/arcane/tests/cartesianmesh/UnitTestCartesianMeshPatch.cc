// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UnitTestCartesianMeshPatch.cc                               (C) 2000-2026 */
/*                                                                           */
/* Service de test des vues cartésiennes sur les patchs.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IMesh.h"

#include "arcane/cea/ICartesianMesh.h"
#include "arcane/cea/ICartesianMeshPatch.h"
#include "arcane/cea/CellDirectionMng.h"
#include "arcane/cea/FaceDirectionMng.h"
#include "arcane/cea/NodeDirectionMng.h"
#include "arcane/cea/CartesianConnectivity.h"

#include "arcane/ItemPrinter.h"
#include "arcane/SimpleSVGMeshExporter.h"
#include "arcane/Directory.h"

#include "arcane/tests/cartesianmesh/UnitTestCartesianMeshPatch_axl.h"

#include <vector>

#define CORRECT_UID

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*!
 * \brief
 * Classe de tests unitaires pour les vues cartésiennes sur les patchs
 */
/*---------------------------------------------------------------------------*/

class UnitTestCartesianMeshPatchService
: public ArcaneUnitTestCartesianMeshPatchObject
{
 public:
  explicit UnitTestCartesianMeshPatchService(const ServiceBuildInfo &sbi);
  virtual ~UnitTestCartesianMeshPatchService() {}

  const Arcane::String getImplName() const { return serviceInfo()->localName(); }

  /*
   * Operations a faire avant l'ensemble des tests du service
   */

  void setUpForClass();

  /*
   * Operations a faire apres l'ensemble des tests du service
   */

  void tearDownForClass();

  /*
   * Operations a faire avant chaque test du service
   */

  void setUp();

  /*
   * Operations a faire apres chaque test du service
   */

  void tearDown();

  /*
   * Test sur les mailles par niveau de raffinement et leurs parents
   */
  void testCartesianMeshPatchCellsAndParents();

  /*
   * Test vue cartésienne sur les mailles
   */
  void testCartesianMeshPatchCellDirectionMng();

  /*
   * Test vue cartésienne sur les faces
   */
  void testCartesianMeshPatchFaceDirectionMng();

  /*
   * Test vue cartésienne sur les noeuds
   */
  void testCartesianMeshPatchNodeDirectionMng();

  /*
   * Test connectivité cartésienne maille -> noeud et noeud -> maille
   */
  void testCartesianMeshPatchCartesianConnectivity();

 private:

  // Pointeur sur le maillage cartésien contenant les vues
  Arcane::ICartesianMesh* m_cartesian_mesh = nullptr;

  // Tableaux de mailles et parents par niveau
  std::vector<std::vector<Arcane::Int64>> m_lvl_cell_uid;
  std::vector<std::vector<Arcane::Int64>> m_lvl_cell_p_uid;
  std::vector<std::vector<Arcane::Int64>> m_lvl_cell_tp_uid;

  // Tableaux des vues cartésiennes sur les mailles par patch
  std::vector<std::vector<Arcane::Int64>> m_patch_cell_uid;
  std::vector<std::vector<std::vector<Arcane::Int64>>> m_patch_celldir_next_uid;
  std::vector<std::vector<std::vector<Arcane::Int64>>> m_patch_celldir_prev_uid;
  std::vector<std::vector<std::vector<Arcane::Int64>>> m_patch_cellfacedir_next_uid;
  std::vector<std::vector<std::vector<Arcane::Int64>>> m_patch_cellfacedir_prev_uid;
  std::vector<std::vector<Arcane::Int64>> m_patch_cellnode_upper_right_uid;
  std::vector<std::vector<Arcane::Int64>> m_patch_cellnode_upper_left_uid;
  std::vector<std::vector<Arcane::Int64>> m_patch_cellnode_lower_right_uid;
  std::vector<std::vector<Arcane::Int64>> m_patch_cellnode_lower_left_uid;

  // Tableaux des vues cartésiennes sur les faces par patch
  std::vector<std::vector<std::vector<Arcane::Int64>>> m_patch_facedir_uid{};
  std::vector<std::vector<std::vector<Arcane::Int64>>> m_patch_facedir_next_cell_uid{};
  std::vector<std::vector<std::vector<Arcane::Int64>>> m_patch_facedir_prev_cell_uid{};

  // Tableaux des vues cartésiennes sur les noeuds par patch
  std::vector<std::vector<Arcane::Int64>> m_patch_node_uid;
  std::vector<std::vector<std::vector<Arcane::Int64>>> m_patch_nodedir_next_uid;
  std::vector<std::vector<std::vector<Arcane::Int64>>> m_patch_nodedir_prev_uid;

	// Tableaux des connectivités noeuds -> mailles par patch
  std::vector<std::vector<Arcane::Int64>> m_patch_nodecell_upper_right_uid;
  std::vector<std::vector<Arcane::Int64>> m_patch_nodecell_upper_left_uid;
  std::vector<std::vector<Arcane::Int64>> m_patch_nodecell_lower_right_uid;
  std::vector<std::vector<Arcane::Int64>> m_patch_nodecell_lower_left_uid;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonction permettant de récupérer la maille en bas à gauche d'un noeud.
 *
 * \param[in] t_node : Noeud pour lequel on cherche la maille en bas à gauche.
 * \param[in] t_cm_patch : Vue cartésienne du patch dans lequel sont le noeud et la maille en bas à gauche.
 * \param[in] t_lvl : Niveau de raffinement de la maille en bas à gauche recherchée.
 * \return Maille en bas à gauche du noeud.
 */
/*---------------------------------------------------------------------------*/
inline Cell lowerLeft(const Node& t_node, ICartesianMeshPatch* t_cm_patch, const Int32 t_lvl)
{
  CellDirectionMng cell_dmy{t_cm_patch->cellDirection(MD_DirY)};
  for (const Cell& cell : t_node.cells()) {
    if (cell.level() == t_lvl) {
      const DirCellNode& dir_cell_nodey{cell_dmy.cellNode(cell)};
      const Node& node_upper_right{dir_cell_nodey.nextRight()};
      if (node_upper_right == t_node) {
        return cell;
      }
    }
  }
  return Cell{};
}

/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonction permettant de récupérer la maille en bas à droite d'un noeud.
 *
 * \param[in] t_node : Noeud pour lequel on cherche la maille en bas à droite.
 * \param[in] t_cm_patch : Vue cartésienne du patch dans lequel sont le noeud et la maille en bas à droite.
 * \param[in] t_lvl : Niveau de raffinement de la maille en bas à droite recherchée.
 * \return Maille en bas à droite du noeud.
 */
/*---------------------------------------------------------------------------*/
inline Cell lowerRight(const Node& t_node, ICartesianMeshPatch* t_cm_patch, const Int32 t_lvl)
{
  CellDirectionMng cell_dmy{t_cm_patch->cellDirection(MD_DirY)};
  for (const Cell& cell : t_node.cells()) {
    if (cell.level() == t_lvl) {
      const DirCellNode& dir_cell_nodey{cell_dmy.cellNode(cell)};
      const Node& node_upper_left{dir_cell_nodey.nextLeft()};
      if (node_upper_left == t_node) {
        return cell;
      }
    }
  }
  return Cell{};
}

/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonction permettant de récupérer la maille en haut à gauche d'un noeud.
 *
 * \param[in] t_node : Noeud pour lequel on cherche la maille en haut à gauche.
 * \param[in] t_cm_patch : Vue cartésienne du patch dans lequel sont le noeud et la maille en haut à gauche.
 * \param[in] t_lvl : Niveau de raffinement de la maille en haut à gauche recherchée.
 * \return Maille en haut à gauche du noeud.
 */
/*---------------------------------------------------------------------------*/
inline Cell upperLeft(const Node& t_node, ICartesianMeshPatch* t_cm_patch, const Int32 t_lvl)
{
  CellDirectionMng cell_dmy{t_cm_patch->cellDirection(MD_DirY)};
  for (const Cell& cell : t_node.cells()) {
    if (cell.level() == t_lvl) {
      const DirCellNode& dir_cell_nodey{cell_dmy.cellNode(cell)};
      const Node& node_lower_right{dir_cell_nodey.previousRight()};
      if (node_lower_right == t_node) {
        return cell;
      }
    }
  }
  return Cell{};
}

/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonction permettant de récupérer la maille en haut à droite d'un noeud.
 *
 * \param[in] t_node : Noeud pour lequel on cherche la maille en haut à droite.
 * \param[in] t_cm_patch : Vue cartésienne du patch dans lequel sont le noeud et la maille en haut à droite.
 * \param[in] t_lvl : Niveau de raffinement de la maille en haut à droite recherchée.
 * \return Maille en haut à droite du noeud.
 */
/*---------------------------------------------------------------------------*/
inline Cell upperRight(const Node& t_node, ICartesianMeshPatch* t_cm_patch, const Int32 t_lvl)
{
  CellDirectionMng cell_dmy{t_cm_patch->cellDirection(MD_DirY)};
  for (const Cell& cell : t_node.cells()) {
    if (cell.level() == t_lvl) {
      const DirCellNode& dir_cell_nodey{cell_dmy.cellNode(cell)};
      const Node& node_lower_left{dir_cell_nodey.previousLeft()};
      if (node_lower_left == t_node) {
        return cell;
      }
    }
  }
  return Cell{};
}

/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonction permettant de récupérer la maille devant une face.
 *
 * \param[in] t_dir_face : Info sur les mailles avant et après la face.
 * \param[in] t_lvl : Niveau de raffinement de la maille devant la face recherchée.
 * \return Maille devant la face.
 */
/*---------------------------------------------------------------------------*/
inline Cell prevCell(const DirFace& t_dir_face, const Int32 t_lvl)
{
  const Cell& cell_prev{t_dir_face.previousCell()};
  if (cell_prev.level() == t_lvl) {
    return cell_prev;
  } else {
    return Cell{};
  }
}

/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonction permettant de récupérer la maille derrière une face.
 *
 * \param[in] t_dir_face : Info sur les mailles avant et après la face.
 * \param[in] t_lvl : Niveau de raffinement de la maille derrière la face recherchée.
 * \return Maille derrière la face.
 */
/*---------------------------------------------------------------------------*/
inline Cell nextCell(const DirFace& t_dir_face, const Int32 t_lvl)
{
  const Cell& cell_next{t_dir_face.nextCell()};
  if (cell_next.level() == t_lvl) {
    return cell_next;
  } else {
    return Cell{};
  }
}

/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonction permettant de récupérer la maille précédente.
 *
 * \param[in] t_dir_cell : Info sur les mailles précédente et suivante.
 * \param[in] t_lvl : Niveau de raffinement de la maille précédente recherchée.
 * \return Maille précédente.
 */
/*---------------------------------------------------------------------------*/
inline Cell prev(const DirCell& t_dir_cell, const Int32 t_lvl)
{
  const Cell& cell_prev{t_dir_cell.previous()};
  if (cell_prev.level() == t_lvl) {
    return cell_prev;
  } else {
    return Cell{};
  }
}

/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonction permettant de récupérer la maille suivante.
 *
 * \param[in] t_dir_cell : Info sur les mailles précédente et suivante.
 * \param[in] t_lvl : Niveau de raffinement de la maille suivante recherchée.
 * \return Maille suivante.
 */
/*---------------------------------------------------------------------------*/
inline Cell next(const DirCell& t_dir_cell, const Int32 t_lvl)
{
  const Cell& cell_next{t_dir_cell.next()};
  if (cell_next.level() == t_lvl) {
    return cell_next;
  } else {
    return Cell{};
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UnitTestCartesianMeshPatchService::
UnitTestCartesianMeshPatchService(const ServiceBuildInfo& sbi)
: ArcaneUnitTestCartesianMeshPatchObject(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*!
 * \brief
 * Actions a effectuer pour tous les tests
 */
/*---------------------------------------------------------------------------*/
void UnitTestCartesianMeshPatchService::
setUpForClass()
{
  // Récupération du pointeur sur le maillage cartésien, raffinement par patch et création des vues
  m_cartesian_mesh = ICartesianMesh::getReference(this->mesh());
  m_cartesian_mesh->refinePatch2D(Real2(0.2, 0.2), Real2(0.4, 0.4));
  m_cartesian_mesh->refinePatch2D(Real2(0.6, 0.6), Real2(0.2, 0.4));
  m_cartesian_mesh->refinePatch2D(Real2(0.3, 0.4), Real2(0.1, 0.8));
  m_cartesian_mesh->refinePatch2D(Real2(0.3, 0.6), Real2(0.05, 0.1));
  m_cartesian_mesh->computeDirections();

  // Sauvegarde chaque patch au format SVG pour les visualer
  {
    Integer nb_patch = m_cartesian_mesh->nbPatch();
    Directory export_dir(subDomain()->exportDirectory());
    for( Integer i=0; i<nb_patch; ++i ){
      ICartesianMeshPatch* patch = m_cartesian_mesh->patch(i);
      String filename = export_dir.file(String::format("Patch{0}.svg",i));
      std::ofstream ofile(filename.localstr());
      SimpleSVGMeshExporter writer(ofile);
      writer.write(patch->cells());
    }
  }

  // Maillage :
  //
  // Patch 1 : +
  // Patch 2 : o
  // Patch 3 : x
  // Patch 4 : =
  //
  //  ------------------------------xxxxxxxxxxxxx-------------------ooooooooooooooooooooo--------------------
  //  |         |         |         x     |     x         |         o    |    |    |    o         |         |
  //  |         |         |         x 104 | 105 x         |         o 92 | 93 | 96 | 97 o         |         |
  //  |         |         |         x     |     x         |         o    |    |    |    o         |         |
  //  |    40   |    41   |    42   x-----43----x    44   |    45   o----46---|----47---o    48   |    49   |
  //  |         |         |         x     |     x         |         o    |    |    |    o         |         |
  //  |         |         |         x 102 | 103 x         |         o 90 | 91 | 94 | 95 o         |         |
  //  |         |         |         x     |     x         |         o    |    |    |    o         |         |
  //  ------------------------------x-----------x-------------------o----|---------|----o--------------------
  //  |         |         |         x     |     x         |         o    |    |    |    o         |         |
  //  |         |         |         x 100 | 101 x         |         o 84 | 85 | 88 | 89 o         |         |
  //  |         |         |         x     |     x         |         o    |    |    |    o         |         |
  //  |    30   |    31   |    32   ======33----x    34   |    35   o----36---|----37---o    38   |    39   |
  //  |         |         |         =28|29=     x         |         o    |    |    |    o         |         |
  //  |         |         |         =--98-=  99 x         |         o 82 | 83 | 86 | 87 o         |         |
  //  |         |         |         =26|27=     x         |         o    |    |    |    o         |         |
  //  --------------------++++++++++=======+++++x+++++++++++++++++++ooooooooooooooooooooo--------------------
  //  |         |         +    |    x16|17|20|21x    |    |    |    +         |         |         |         |
  //  |         |         + 68 | 69 x--72-|--73-x 76 | 77 | 80 | 81 +         |         |         |         |
  //  |         |         +    |    x14|15|18|19x    |    |    |    +         |         |         |         |
  //  |    20   |    21   +----22---x-----23----x----24---|----25---+    26   |    27   |    28   |    29   |
  //  |         |         +    |    x08|09|12|13x    |    |    |    +         |         |         |         |
  //  |         |         + 66 | 67 x--70-|--71-x 74 | 75 | 78 | 79 +         |         |         |         |
  //  |         |         +    |    x06|07|10|11x    |    |    |    +         |         |         |         |
  //  --------------------+---------xxxxxxxxxxxxx-------------------+----------------------------------------
  //  |         |         +    |    |     |     |    |    |    |    +         |         |         |         |
  //  |         |         + 52 | 53 |  56 |  57 | 60 | 61 | 64 | 65 +         |         |         |         |
  //  |         |         +    |    |     |     |    |    |    |    +         |         |         |         |
  //  |    10   |    11   +----12---|-----13----|----14---|----15---+    16   |    17   |    18   |    19   |
  //  |         |         +    |    |     |     |    |    |    |    +         |         |         |         |
  //  |         |         + 50 | 51 |  54 |  55 | 58 | 59 | 62 | 63 +         |         |         |         |
  //  |         |         +    |    |     |     |    |    |    |    +         |         |         |         |
  //  --------------------+++++++++++++++++++++++++++++++++++++++++++----------------------------------------
  //  |         |         |         |           |         |         |         |         |         |         |
  //  |         |         |         |           |         |         |         |         |         |         |
  //  |         |         |         |           |         |         |         |         |         |         |
  //  |    0    |    1    |    2    |     3     |    4    |    5    |     6   |    7    |    8    |    9    |
  //  |         |         |         |           |         |         |         |         |         |         |
  //  |         |         |         |           |         |         |         |         |         |         |
  //  |         |         |         |           |         |         |         |         |         |         |
  //  -------------------------------------------------------------------------------------------------------

  // clang-format off

  // Tableaux de mailles et parents par niveau :
  {
    // Level 0
		{
    	m_lvl_cell_uid.push_back(
				{  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
				  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
					20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
					30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
					40,  41,  42,  43,  44,  45,  46,  47,  48,  49});

			m_lvl_cell_p_uid.push_back({});
			m_lvl_cell_tp_uid.push_back({});
		}

		// Level 1
		{
			m_lvl_cell_uid.push_back(
				{ 50,  51,  52,  53,    54,  55,  56,  57,    58,  59,  60,  61,    62,  63,  64,  65,
					66,  67,  68,  69,    70,  71,  72,  73,    74,  75,  76,  77,    78,  79,  80,  81,

					82,  83,  84,  85,    86,  87,  88,  89,
					90,  91,  92,  93,    94,  95,  96,  97,

					98,  99, 100, 101,
				 102, 103, 104, 105,

				 122, 123, 124, 125});

			const std::vector<Int64> cell_p_uid{
				{ 12,  12,  12,  12,    13,  13,  13,  13,    14,  14,  14,  14,    15,  15,  15,  15,
					22,  22,  22,  22,    23,  23,  23,  23,    24,  24,  24,  24,    25,  25,  25,  25,

					36,  36,  36,  36,    37,  37,  37,  37,
					46,  46,  46,  46,    47,  47,  47,  47,

					33,  33,  33,  33,
					43,  43,  43,  43,

					32,  32,  32,  32}};
			m_lvl_cell_p_uid.push_back(cell_p_uid);
			m_lvl_cell_tp_uid.push_back(cell_p_uid);
		}

		// Level 2
		{
			m_lvl_cell_uid.push_back(
				{106, 107, 108, 109,   110, 111, 112, 113,
				 114, 115, 116, 117,   118, 119, 120, 121,

				 126, 127, 128, 129});

			m_lvl_cell_p_uid.push_back(
				{ 70,  70,  70,  70,    71,  71,  71,  71,
					72,  72,  72,  72,    73,  73,  73,  73,

					98,  98,  98,  98});

			m_lvl_cell_tp_uid.push_back(
				{ 23,  23,  23,  23,     23,  23,  23,  23,
					23,  23,  23,  23,     23,  23,  23,  23,

					33,  33,  33,  33});
		}
  }

  // Tableaux des vues cartésiennes sur les mailles par patch
	{
		// Patch 0 :
		{
			//  55<--86->-56<--90->-57<--93->-58<--96->-59<--99->-60<-102->-61<-105->-62<-108->-63<-111->-64<-114->-65
			//  |         |         |         |         |         |         |         |         |         |         |
			//  87   40   85   41   89   42   92   43   95   44   98   45  101   46  104   47  107   48  110   49  113
			//  |         |         |         |         |         |         |         |         |         |         |
			//  44<--84->-45<--88->-46<--91->-47<--94->-48<--97->-49<-100->-50<-103->-51<-106->-52<-109->-53<-112->-54
			//  |         |         |         |         |         |         |         |         |         |         |
			//  65   30   64   31   67   32   69   33   71   34   73   35   75   36   77   37   79   38   81   39   83
			//  |         |         |         |         |         |         |         |         |         |         |
			//  33<--63->-34<--66->-35<--68->-36<--70->-37<--72->-38<--74->-39<--76->-40<--78->-41<--80->-42<--82->-43
			//  |         |         |         |         |         |         |         |         |         |         |
			//  44   20   43   21   46   22   48   23   50   24   52   25   54   26   56   27   58   28   60   29   62
			//  |         |         |         |         |         |         |         |         |         |         |
			//  22<--42->-23<--45->-24<--47->-25<--49->-26<--51->-27<--53->-28<--55->-29<--57->-30<--59->-31<--61->-32
			//  |         |         |         |         |         |         |         |         |         |         |
			//  23   10   22   11   25   12   27   13   29   14   31   15   33   16   35   17   37   18   39   19   41
			//  |         |         |         |         |         |         |         |         |         |         |
			//  11<--21->-12<--24->-13<--26->-14<--28->-15<--30->-16<--32->-17<--34->-18<--36->-19<--38->-20<--40->-21
			//  |         |         |         |         |         |         |         |         |         |         |
			//  2    0    1    1    4    2    6    3    8    4    10   5    12    6   14   7    16   8    18   9    20
			//  |         |         |         |         |         |         |         |         |         |         |
			//  0-<--0-->-1-<--3-->-2-<--5-->-3-<--7-->-4-<--9-->-5-<--11->-6-<--13->-7-<--15->-8-<--17->-9-<--19->-10

			m_patch_cell_uid.push_back(
				{  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
					10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
					20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
					30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
					40,  41,  42,  43,  44,  45,  46,  47,  48,  49});

			m_patch_celldir_next_uid.push_back(
				{ // X
					{  1,   2,   3,   4,   5,   6,   7,   8,   9,  -1,
					  11,  12,  13,  14,  15,  16,  17,  18,  19,  -1,
					  21,  22,  23,  24,  25,  26,  27,  28,  29,  -1,
					  31,  32,  33,  34,  35,  36,  37,  38,  39,  -1,
					  41,  42,  43,  44,  45,  46,  47,  48,  49,  -1},
					// Y
					{ 10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
						20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
						30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
						40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
						-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1}});

			m_patch_celldir_prev_uid.push_back(
				{	// X
					{ -1,   0,   1,   2,   3,   4,   5,   6,   7,   8,
						-1,  10,  11,  12,  13,  14,  15,  16,  17,  18,
						-1,  20,  21,  22,  23,  24,  25,  26,  27,  28,
						-1,  30,  31,  32,  33,  34,  35,  36,  37,  38,
						-1,  40,  41,  42,  43,  44,  45,  46,  47,  48},
					// Y
					{ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
						0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
						10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
						20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
						30,  31,  32,  33,  34,  35,  36,  37,  38,  39}});

			m_patch_cellfacedir_next_uid.push_back(
				{	// X
					{  1,   4,   6,   8,  10,  12,  14,  16,  18,  20,
						22,  25,  27,  29,  31,  33,  35,  37,  39,  41,
						43,  46,  48,  50,  52,  54,  56,  58,  60,  62,
						64,  67,  69,  71,  73,  75,  77,  79,  81,  83,
						85,  89,  92,  95,  98, 101, 104, 107, 110, 113},
					// Y
					{ 21,  24,  26,  28,  30,  32,  34,  36,  38,  40,
						42,  45,  47,  49,  51,  53,  55,  57,  59,  61,
						63,  66,  68,  70,  72,  74,  76,  78,  80,  82,
						84,  88,  91,  94,  97, 100, 103, 106, 109, 112,
						86,  90,  93,  96,  99, 102, 105, 108, 111, 114}});

			m_patch_cellfacedir_prev_uid.push_back(
				{	// X
					{  2,   1,   4,   6,   8,  10,  12,  14,  16,  18,
						23,  22,  25,  27,  29,  31,  33,  35,  37,  39,
						44,  43,  46,  48,  50,  52,  54,  56,  58,  60,
						65,  64,  67,  69,  71,  73,  75,  77,  79,  81,
						87,  85,  89,  92,  95,  98, 101, 104, 107, 110},
					// Y
					{  0,   3,   5,   7,   9,  11,  13,  15,  17,  19,
						21,  24,  26,  28,  30,  32,  34,  36,  38,  40,
						42,  45,  47,  49,  51,  53,  55,  57,  59,  61,
						63,  66,  68,  70,  72,  74,  76,  78,  80,  82,
						84,  88,  91,  94,  97, 100, 103, 106, 109, 112}});

			m_patch_cellnode_upper_right_uid.push_back(
				{ 12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
					23,  24,  25,  26,  27,  28,  29,  30,  31,  32,
					34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
					45,  46,  47,  48,  49,  50,  51,  52,  53,  54,
					56,  57,  58,  59,  60,  61,  62,  63,  64,  65});

			m_patch_cellnode_upper_left_uid.push_back(
				{ 11,  12,  13,  14,  15,  16,  17,  18,  19,  20,
					22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
					33,  34,  35,  36,  37,  38,  39,  40,  41,  42,
					44,  45,  46,  47,  48,  49,  50,  51,  52,  53,
					55,  56,  57,  58,  59,  60,  61,  62,  63,  64});

			m_patch_cellnode_lower_right_uid.push_back(
				{  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,
					12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
					23,  24,  25,  26,  27,  28,  29,  30,  31,  32,
					34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
					45,  46,  47,  48,  49,  50,  51,  52,  53,  54});

			m_patch_cellnode_lower_left_uid.push_back(
				{  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
					11,  12,  13,  14,  15,  16,  17,  18,  19,  20,
					22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
					33,  34,  35,  36,  37,  38,  39,  40,  41,  42,
					44,  45,  46,  47,  48,  49,  50,  51,  52,  53});
		}

		// Patch 1 :
		{
			//  35<-211->106<-217->-36<-229->112<-233->-37<-245->118<-249->-38<-261->124<-265->-39
			//  |         |         |         |         |         |         |         |         |
			// 213   68  209   69  215   72  227   73  231   76  243   77  247   80  259   81  263
			//  |         |         |         |         |         |         |         |         |
			// 102<-201->100<-207->104<-221->108<-225->110<-237->114<-241->116<-253->120<-257->122
			//  |         |         |         |         |         |         |         |         |
			// 203   66  199   67  205   70  219   71  223   74  235   75  239   78  251   79  255
			//  |         |         |         |         |         |         |         |         |
			//  24<-131->-74<-137->-25<-153->-82<-157->-26<-173->-90<-177->-27<-193->-98<-197->-28
			//  |         |         |         |         |         |         |         |         |
			// 133   52  129   53  135   56  151   57  155   60  171   61  175   64  191   65  195
			//  |         |         |         |         |         |         |         |         |
			//  70<-119->-68<-127->-72<-143->-78<-149->-80<-163->-86<-169->-88<-183->-94<-189->-96
			//  |         |         |         |         |         |         |         |         |
			// 121   50  117   51  125   54  141   55  147   58  161   59  167   62  181   63  187
			//  |         |         |         |         |         |         |         |         |
			//  13<-115->-66<-123->-14<-139->-76<-145->-15<-159->-84<-165->-16<-179->-92<-185->-17

			m_patch_cell_uid.push_back(
				{ 50,  51,  52,  53,    54,  55,  56,  57,    58,  59,  60,  61,    62,  63,  64,  65,
					66,  67,  68,  69,    70,  71,  72,  73,    74,  75,  76,  77,    78,  79,  80,  81});

			m_patch_celldir_next_uid.push_back(
				{ // X
					{ 51,  54,  53,  56,    55,  58,  57,  60,    59,  62,  61,  64,    63,  -1,  65,  -1,
						67,  70,  69,  72,    71,  74,  73,  76,    75,  78,  77,  80,    79,  -1,  81,  -1},
					// Y
					{ 52,  53,  66,  67,    56,  57,  70,  71,    60,  61,  74,  75,    64,  65,  78,  79,
						68,  69,  -1,  -1,    72,  73,  -1,  -1,    76,  77,  -1,  -1,    80,  81,  -1,  -1}});

			m_patch_celldir_prev_uid.push_back(
				{	// X
					{ -1,  50,  -1,  52,    51,  54,  53,  56,    55,  58,  57,  60,    59,  62,  61,  64,
						-1,  66,  -1,  68,    67,  70,  69,  72,    71,  74,  73,  76,    75,  78,  77,  80},
					// Y
					{ -1,  -1,  50,  51,    -1,  -1,  54,  55,    -1,  -1,  58,  59,    -1,  -1,  62,  63,
						52,  53,  66,  67,    56,  57,  70,  71,    60,  61,  74,  75,    64,  65,  78,  79}});

			m_patch_cellfacedir_next_uid.push_back(
				{ // X
					{117, 125, 129, 135,   141, 147, 151, 155,   161, 167, 171, 175,   181, 187, 191, 195,
					 199, 205, 209, 215,   219, 223, 227, 231,   235, 239, 243, 247,   251, 255, 259, 263},
					// Y
					{119, 127, 131, 137,   143, 149, 153, 157,   163, 169, 173, 177,   183, 189, 193, 197,
					 201, 207, 211, 217,   221, 225, 229, 233,   237, 241, 245, 249,   253, 257, 261, 265}});

			m_patch_cellfacedir_prev_uid.push_back(
				{ // X
					{121, 117, 133, 129,   125, 141, 135, 151,   147, 161, 155, 171,   167, 181, 175, 191,
					 203, 199, 213, 209,   205, 219, 215, 227,   223, 235, 231, 243,   239, 251, 247, 259},
					// Y
					{115, 123, 119, 127,   139, 145, 143, 149,   159, 165, 163, 169,   179, 185, 183, 189,
					 131, 137, 201, 207,   153, 157, 221, 225,   173, 177, 237, 241,   193, 197, 253, 257}});

			m_patch_cellnode_upper_right_uid.push_back(
				{ 68,  72,  74,  25,    78,  80,  82,  26,    86,  88,  90,  27,    94,  96,  98,  28,
				 100, 104, 106,  36,   108, 110, 112,  37,   114, 116, 118,  38,   120, 122, 124,  39});

			m_patch_cellnode_upper_left_uid.push_back(
				{ 70,  68,  24,  74,    72,  78,  25,  82,    80,  86,  26,  90,    88,  94,  27,  98,
				 102, 100,  35, 106,   104, 108,  36, 112,   110, 114,  37, 118,   116, 120,  38, 124});

			m_patch_cellnode_lower_right_uid.push_back(
				{ 66,  14,  68,  72,    76,  15,  78,  80,    84,  16,  86,  88,    92,  17,  94,  96,
					74,  25, 100, 104,    82,  26, 108, 110,    90,  27, 114, 116,    98,  28, 120, 122});

			m_patch_cellnode_lower_left_uid.push_back(
				{ 13,  66,  70,  68,    14,  76,  72,  78,    15,  84,  80,  86,    16,  92,  88,  94,
					24,  74, 102, 100,    25,  82, 104, 108,    26,  90, 110, 114,    27,  98, 116, 120});
		}

		// Patch 2 :
		{
			//  61<-322->149<-328->-62<-340->155<-344->-63
			//  |         |         |         |         |
			// 324   92  320   93  326   96  338   97  342
			//  |         |         |         |         |
			// 145<-312->143<-318->147<-332->151<-336->153
			//  |         |         |         |         |
			// 314   90  310   91  316   94  330   95  334
			//  |         |         |         |         |
			//  50<-282->133<-288->-51<-304->141<-308->-52
			//  |         |         |         |         |
			// 284   84  280   85  286   88  302   89  306
			//  |         |         |         |         |
			// 129<-270->127<-278->131<-294->137<-300->139
			//  |         |         |         |         |
			// 272   82  268   83  276   86  292   87  298
			//  |         |         |         |         |
			//  39<-266->125<-274->-40<-290->135<-296->-41

			m_patch_cell_uid.push_back(
        { 82,  83,  84,  85,    86,  87,  88,  89,
          90,  91,  92,  93,    94,  95,  96,  97});

			m_patch_celldir_next_uid.push_back(
				{ // X
					{ 83,  86,  85,  88,    87,  -1,  89,  -1,
						91,  94,  93,  96,    95,  -1,  97,  -1},
					// Y
					{ 84,  85,  90,  91,    88,  89,  94,  95,
					  92,  93,  -1,  -1,    96,  97,  -1,  -1}});

			m_patch_celldir_prev_uid.push_back(
				{	// X
					{ -1,  82,  -1,  84,    83,  86,  85,  88,
						-1,  90,  -1,  92,    91,  94,  93,  96},
					// Y
					{ -1,  -1,  82,  83,    -1,  -1,  86,  87,
						84,  85,  90,  91,    88,  89,  94,  95}});

			m_patch_cellfacedir_next_uid.push_back(
				{ // X
					{268, 276, 280, 286,   292, 298, 302, 306,
					 310, 316, 320, 326,   330, 334, 338, 342},
					// Y
					{270, 278, 282, 288,   294, 300, 304, 308,
					 312, 318, 322, 328,   332, 336, 340, 344}});

			m_patch_cellfacedir_prev_uid.push_back(
				{ // X
					{272, 268, 284, 280,   276, 292, 286, 302,
					 314, 310, 324, 320,   316, 330, 326, 338},
					// Y
					{266, 274, 270, 278,   290, 296, 294, 300,
					 282, 288, 312, 318,   304, 308, 332, 336}});

			m_patch_cellnode_upper_right_uid.push_back(
        {127, 131, 133,  51,   137, 139, 141,  52,
         143, 147, 149,  62,   151, 153, 155,  63});

			m_patch_cellnode_upper_left_uid.push_back(
        {129, 127,  50, 133,   131, 137,  51, 141,
         145, 143,  61, 149,   147, 151,  62, 155});

			m_patch_cellnode_lower_right_uid.push_back(
				{125,  40, 127, 131,   135,  41, 137, 139,
				 133,  51, 143, 147,   141,  52, 151, 153});

			m_patch_cellnode_lower_left_uid.push_back(
        { 39, 125, 129, 127,    40, 135, 131, 137,
          50, 133, 145, 143,    51, 141, 147, 151});
		}

		// Patch 3 :
		{
			//  58<------377------>170<------383------>-59
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 379       104       375       105       381
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 166<------367------>164<------373------>168
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 369       102       365       103       371
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  47<------357------>162<------363------>-48
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 359       100       355       101       361
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 158<------347------>156<------353------>160
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 349        98       345        99       351
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  36<------229------>112<------233------>-37
			//  36<-441->196<-447->112<-459->202<-463->-37
			//  |         |         |         |         |
			// 443   116 439  117  445  120  457  121  461
			//  |         |         |         |         |
			// 192<-431->190<-437->194<-451->198<-455->200
			//  |         |         |         |         |
			// 433  114  429  115  435  118  449  119  453
			//  |         |         |         |         |
			// 104<-401->180<-407->108<-423->188<-427->110
			//  |         |         |         |         |
			// 403  108  399  109  405  112  421  113  425
			//  |         |         |         |         |
			// 176<-389->174<-397->178<-413->184<-419->186
			//  |         |         |         |         |
			// 391  106  387  107  395  110  411  111  417
			//  |         |         |         |         |
			//  25<-385->172<-393->-82<-409->182<-415->-26

			m_patch_cell_uid.push_back(
        { 98,  99, 100, 101,
         102, 103, 104, 105,
         106, 107, 108, 109,   110, 111, 112, 113,
         114, 115, 116, 117,   118, 119, 120, 121});

			m_patch_celldir_next_uid.push_back(
				{ // X
					{ 99,  -1, 101,  -1,
					 103,  -1, 105,  -1,
					 107, 110, 109, 112,   111,  -1, 113,  -1,
					 115, 118, 117, 120,   119,  -1, 121,  -1},
					// Y
					{100, 101, 102, 103,
					 104, 105,  -1,  -1,
					 108, 109, 114, 115,   112, 113, 118, 119,
					#ifdef CORRECT_UID
					 116, 117,  -1,  -1,   120, 121,  -1,  -1}});
					#else
					 116, 117,  -1,  -1,   120, 121,  99,  99}});  // error 99 -> -1 or -1 -> 98
					#endif

			m_patch_celldir_prev_uid.push_back(
				{	// X
					{ -1,  98,  -1, 100,
						-1, 102,  -1, 104,
						-1, 106,  -1, 108,   107, 110, 109, 112,
						-1, 114,  -1, 116,   115, 118, 117, 120},
					// Y
					{ -1,  -1,  98,  99,
					 100, 101, 102, 103,
						-1,  -1, 106, 107,    -1,  -1, 110, 111,
					 108, 109, 114, 115,   112, 113, 118, 119}});

			m_patch_cellfacedir_next_uid.push_back(
				{ // X
					{345, 351, 355, 361,
					 365, 371, 375, 381,
					 387, 395, 399, 405,   411, 417, 421, 425,
					 429, 435, 439, 445,   449, 453, 457, 461},
					// Y
					{347, 353, 357, 363,
					 367, 373, 377, 383,
					 389, 397, 401, 407,   413, 419, 423, 427,
					 431, 437, 441, 447,   451, 455, 459, 463}});

			m_patch_cellfacedir_prev_uid.push_back(
				{ // X
					{349, 345, 359, 355,
					 369, 365, 379, 375,
					 391, 387, 403, 399,   395, 411, 405, 421,
					 433, 429, 443, 439,   435, 449, 445, 457},
					// Y
					{229, 233, 347, 353,
					 357, 363, 367, 373,
					 385, 393, 389, 397,   409, 415, 413, 419,
					 401, 407, 431, 437,   423, 427, 451, 455}});

			m_patch_cellnode_upper_right_uid.push_back(
        {156, 160, 162,  48,
         164, 168, 170,  59,
         174, 178, 180, 108,   184, 186, 188, 110,
         190, 194, 196, 112,   198, 200, 202,  37});

			m_patch_cellnode_upper_left_uid.push_back(
        {158, 156,  47, 162,
         166, 164,  58, 170,
         176, 174, 104, 180,   178, 184, 108, 188,
         192, 190,  36, 196,   194, 198, 112, 202});

			m_patch_cellnode_lower_right_uid.push_back(
        {112,  37, 156, 160,
         162,  48, 164, 168,
         172,  82, 174, 178,   182,  26, 184, 186,
         180, 108, 190, 194,   188, 110, 198, 200});

			m_patch_cellnode_lower_left_uid.push_back(
        { 36, 112, 158, 156,
          47, 162, 166, 164,
          25, 172, 176, 174,    82, 182, 178, 184,
         104, 180, 192, 190,   108, 188, 194, 198});
		}

		// Patch 4 :
		{
			// 158<------492------>215<------498------>156
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 494       128       490       129       496
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 211<------482------>209<------488------>213
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 484       126       480       127       486
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  36<------441------>196<------447------>112

			m_patch_cell_uid.push_back(
        {126, 127, 128, 129});

			m_patch_celldir_next_uid.push_back(
				{ // X
        	{127,  -1, 129,  -1},
					// Y
        	{128, 129,  -1,  -1}});

			m_patch_celldir_prev_uid.push_back(
				{	// X
        	{ -1, 126,  -1, 128},
					// Y
        	{ -1,  -1, 126, 127}});

			m_patch_cellfacedir_next_uid.push_back(
				{ // X
        	{480, 486, 490, 496},
					// Y
        	{482, 488, 492, 498}});

			m_patch_cellfacedir_prev_uid.push_back(
				{ // X
        	{484, 480, 494, 490},
					// Y
        	{441, 447, 482, 488}});

			m_patch_cellnode_upper_right_uid.push_back(
        {209, 213, 215, 156});

			m_patch_cellnode_upper_left_uid.push_back(
        {211, 209, 158, 215});

			m_patch_cellnode_lower_right_uid.push_back(
        {196, 112, 209, 213});

			m_patch_cellnode_lower_left_uid.push_back(
        { 36, 196, 211, 209});
		}
	}

  // Tableaux des vues cartésiennes sur les faces par patch
	{
		// Patch 0
		{
			//  -----86--------90--------93--------96--------99-------102-------105-------108-------111-------113----
			//  |         |         |         |         |         |         |         |         |         |         |
			//  85   40   87   41   89   42   92   43   95   44   98   45  101   46  104   47  107   48  110   49  114
			//  |         |         |         |         |         |         |         |         |         |         |
			//  -----84--------88--------91--------94--------97-------100-------103-------106-------109-------112----
			//  |         |         |         |         |         |         |         |         |         |         |
			//  64   30   65   31   67   32   69   33   71   34   73   35   75   36   77   37   79   38   81   39   83
			//  |         |         |         |         |         |         |         |         |         |         |
			//  -----63--------66--------68--------70--------72--------74--------76--------78--------80--------82----
			//  |         |         |         |         |         |         |         |         |         |         |
			//  43   20   44   21   46   22   48   23   50   24   52   25   54   26   56   27   58   28   60   29   62
			//  |         |         |         |         |         |         |         |         |         |         |
			//  -----42--------45--------47--------49--------51--------53--------55--------57--------59--------61----
			//  |         |         |         |         |         |         |         |         |         |         |
			//  22   10   23   11   25   12   27   13   29   14   31   15   33   16   35   17   37   18   39   19   41
			//  |         |         |         |         |         |         |         |         |         |         |
			//  -----21--------24--------26--------28--------30--------32--------34--------36--------38--------40----
			//  |         |         |         |         |         |         |         |         |         |         |
			//  1    0    2    1    4    2    6    3    8    4    10   5    12    6   14   7    16   8    18   9    20
			//  |         |         |         |         |         |         |         |         |         |         |
			//  -----0---------3---------5---------7---------9---------11--------13--------15--------17--------19----

			m_patch_facedir_uid.push_back(
				{	// X
					{  1,   2,   4,   6,   8,  10,  12,  14,  16,  18,  20,
						22,  23,  25,  27,  29,  31,  33,  35,  37,  39,  41,
						43,  44,  46,  48,  50,  52,  54,  56,  58,  60,  62,
						64,  65,  67,  69,  71,  73,  75,  77,  79,  81,  83,
						85,  87,  89,  92,  95,  98, 101, 104, 107, 110, 113},
					// Y
					{  0,   3,   5,   7,   9,  11,  13,  15,  17,  19,
						21,  24,  26,  28,  30,  32,  34,  36,  38,  40,
						42,  45,  47,  49,  51,  53,  55,  57,  59,  61,
						63,  66,  68,  70,  72,  74,  76,  78,  80,  82,
						84,  86,  88,  90,  91,  93,  94,  96,  97,  99,
					 100, 102, 103, 105, 106, 108, 109, 111, 112, 114}});

			m_patch_facedir_next_cell_uid.push_back(
				{ // X
					{  1,   0,   2,   3,   4,   5,   6,   7,   8,   9,  -1,
						11,  10,  12,  13,  14,  15,  16,  17,  18,  19,  -1,
						21,  20,  22,  23,  24,  25,  26,  27,  28,  29,  -1,
						31,  30,  32,  33,  34,  35,  36,  37,  38,  39,  -1,
						41,  40,  42,  43,  44,  45,  46,  47,  48,  49,  -1},
					// Y
					{  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
						10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
						20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
						30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
						40,  -1,  41,  -1,  42,  -1,  43,  -1,  44,  -1,
						45,  -1,  46,  -1,  47,  -1,  48,  -1,  49,  -1}});

			m_patch_facedir_prev_cell_uid.push_back(
				{	// X
					{  0,  -1,   1,   2,   3,   4,   5,   6,   7,   8,   9,
						10,  -1,  11,  12,  13,  14,  15,  16,  17,  18,  19,
						20,  -1,  21,  22,  23,  24,  25,  26,  27,  28,  29,
						30,  -1,  31,  32,  33,  34,  35,  36,  37,  38,  39,
						40,  -1,  41,  42,  43,  44,  45,  46,  47,  48,  49},
					// Y
					{ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
						0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
						10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
						20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
						30,  40,  31,  41,  32,  42,  33,  43,  34,  44,
						35,  45,  36,  46,  37,  47,  38,  48,  39,  49}});
		}

		// Patch 1
		{
			//  ----211-------217-------229-------233-------245-------249-------261-------265----
			//  |         |         |         |         |         |         |         |         |
			// 213   68  209   69  215   72  227   73  231   76  243   77  247   80  259   81  263
			//  |         |         |         |         |         |         |         |         |
			//  ----201-------207-------221-------225-------237-------241-------253-------257----
			//  |         |         |         |         |         |         |         |         |
			// 203   66  199   67  205   70  219   71  223   74  235   75  239   78  251   79  255
			//  |         |         |         |         |         |         |         |         |
			//  ----131-------137-------153-------157-------173-------177-------193-------197----
			//  |         |         |         |         |         |         |         |         |
			// 133   52  129   53  135   56  151   57  155   60  171   61  175   64  191   65  195
			//  |         |         |         |         |         |         |         |         |
			//  ----119-------127-------143-------149-------163-------169-------183-------189----
			//  |         |         |         |         |         |         |         |         |
			// 121   50  117   51  125   54  141   55  147   58  161   59  167   62  181   63  187
			//  |         |         |         |         |         |         |         |         |
			//  ----115-------123-------139-------145-------159-------165-------179-------185----

			m_patch_facedir_uid.push_back(
				{	// X
					{117, 121, 125, 129, 133, 135, 141, 147, 151, 155, 161, 167, 171, 175, 181, 187, 191, 195,
					 199, 203, 205, 209, 213, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255, 259, 263},
					// Y
					{115, 119, 123, 127, 131, 137,
					 139, 143, 145, 149, 153, 157,
					 159, 163, 165, 169, 173, 177,
					 179, 183, 185, 189, 193, 197,
					 201, 207, 211, 217,
					 221, 225, 229, 233,
					 237, 241, 245, 249,
					 253, 257, 261, 265}});

			m_patch_facedir_next_cell_uid.push_back(
				{	// X
					#ifdef CORRECT_UID
					{ 51,  50,  54,  53,  52,  56,  55,  58,  57,  60,  59,  62,  61,  64,  63,  -1,  65,  -1,
					  67,  66,  70,  69,  68,  72,  71,  74,  73,  76,  75,  78,  77,  80,  79,  -1,  81,  -1},
					#else
					{ 51,  50,  54,  53,  52,  56,  55,  58,  57,  60,  59,  62,  61,  64,  63,  16,  65,  -1,  // error 16 -> -1
					  67,  66,  70,  69,  68,  72,  71,  74,  73,  76,  75,  78,  77,  80,  79,  26,  81,  -1},  // error 26 -> -1
					#endif
					// Y
					{ 50,  52,  51,  53,  66,  67,
					  54,  56,  55,  57,  70,  71,
						58,  60,  59,  61,  74,  75,
						62,  64,  63,  65,  78,  79,
					#ifdef CORRECT_UID
					  68,  69,  -1,  -1,
					  72,  73,  -1,  -1,
					  76,  77,  -1,  -1,
					  80,  81,  -1,  -1}});
					#else
					  68,  69, 122, 123,  // error 122, 123 -> -1
					  72,  73,  98,  99,  // error 98, 99 -> -1
					  76,  77,  34,  34,  // error 34, 34 -> -1
					  80,  81,  35,  35}});  // error 35, 35 -> -1
					#endif

			m_patch_facedir_prev_cell_uid.push_back(
				{	// X
					#ifdef CORRECT_UID
					{ 50,  -1,  51,  52,  -1,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
					  66,  -1,  67,  68,  -1,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81},
					#else
					{ 50,  11,  51,  52,  -1,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  // error 11 -> -1
					  66,  21,  67,  68,  -1,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81},  // error 21 -> -1
					#endif
					// Y
					#ifdef CORRECT_UID
					{ -1,  50,  -1,  51,  52,  53,
					  -1,  54,  -1,  55,  56,  57,
					  -1,  58,  -1,  59,  60,  61,
					  -1,  62,  -1,  63,  64,  65,
					#else
					{  2,  50,   2,  51,  52,  53,  // error 2, 2 -> -1
					   3,  54,   3,  55,  56,  57,  // error 3, 3 -> -1
					   4,  58,   4,  59,  60,  61,  // error 4, 4 -> -1
					   5,  62,   5,  63,  64,  65,  // error 5, 5 -> -1
					#endif
					  66,  67,  68,  69,
					  70,  71,  72,  73,
					  74,  75,  76,  77,
					  78,  79,  80,  81}});
		}

		// Patch 2 :
		{
			//  ----322-------328-------340-------344----
			//  |         |         |         |         |
			// 324   92  320   93  326   96  338   97  342
			//  |         |         |         |         |
			//  ----312-------318-------332-------336----
			//  |         |         |         |         |
			// 314   90  310   91  316   94  330   95  334
			//  |         |         |         |         |
			//  ----282-------288-------304-------308----
			//  |         |         |         |         |
			// 284   84  280   85  286   88  302   89  306
			//  |         |         |         |         |
			//  ----270-------278-------294-------300----
			//  |         |         |         |         |
			// 272   82  268   83  276   86  292   87  298
			//  |         |         |         |         |
			//  ----266-------274-------290-------296----

			m_patch_facedir_uid.push_back(
				{	// X
					{268, 272, 276, 280, 284, 286, 292, 298, 302, 306,
					 310, 314, 316, 320, 324, 326, 330, 334, 338, 342},
					// Y
					{266, 270, 274, 278, 282, 288, 290, 294, 296, 300, 304, 308,
					 312, 318, 322, 328, 332, 336, 340, 344}});

			m_patch_facedir_next_cell_uid.push_back(
				{	// X
					#ifdef CORRECT_UID
					{ 83,  82,  86,  85,  84,  88,  87,  -1,  89,  -1,  // error 38 -> -1
						91,  90,  94,  93,  92,  96,  95,  -1,  97,  -1},  // error 48 -> -1
					#else
					{ 83,  82,  86,  85,  84,  88,  87,  38,  89,  -1,  // error 38 -> -1
						91,  90,  94,  93,  92,  96,  95,  48,  97,  -1},  // error 48 -> -1
					#endif
					// Y
					{ 82,  84,  83,  85,  90,  91,  86,  88,  87,  89,  94,  95,
						92,  93,  -1,  -1,  96,  97,  -1,  -1}});

			m_patch_facedir_prev_cell_uid.push_back(
				{	// X
					#ifdef CORRECT_UID
					{ 82,  -1,  83,  84,  -1,  85,  86,  87,  88,  89,  // error 35 -> -1
						90,  -1,  91,  92,  -1,  93,  94,  95,  96,  97},  // error 45 -> -1
					#else
					{ 82,  35,  83,  84,  -1,  85,  86,  87,  88,  89,  // error 35 -> -1
						90,  45,  91,  92,  -1,  93,  94,  95,  96,  97},  // error 45 -> -1
					#endif
					// Y
					#ifdef CORRECT_UID
					{ -1,  82,  -1,  83,  84,  85,  -1,  86,  -1,  87,  88,  89,  // error 26, 27 -> -1
					#else
					{ 26,  82,  26,  83,  84,  85,  27,  86,  27,  87,  88,  89,  // error 26, 27 -> -1
					#endif
						90,  91,  92,  93,  94,  95,  96,  97}});
		}

		// Patch 3 :
		{
			//  ---------377-----------------383---------
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 379       104       375       105       381
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  ---------367-----------------373---------
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 369       102       365       103       371
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  ---------357-----------------363---------
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 359       100       355       101       361
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  ---------347-----------------353---------
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 349        98       345        99       351
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  ---------229-----------------233---------
			//  ----441-------447-------459-------463----
			//  |         |         |         |         |
			// 443   116 439  117  445  120  457  121  461
			//  |         |         |         |         |
			//  ----431-------437-------451-------455----
			//  |         |         |         |         |
			// 433  114  429  115  435  118  449  119  453
			//  |         |         |         |         |
			//  ----401-------407-------423-------427----
			//  |         |         |         |         |
			// 403  108  399  109  405  112  421  113  425
			//  |         |         |         |         |
			//  ----389-------397-------413-------419----
			//  |         |         |         |         |
			// 391  106  387  107  395  110  411  111  417
			//  |         |         |         |         |
			//  ----385-------393-------409-------415----

			m_patch_facedir_uid.push_back(
				{	// X
					{345, 349, 351, 355, 359, 361,
					 365, 369, 371, 375, 379, 381,
					 387, 391, 395, 399, 403, 405, 411, 417, 421, 425,
					 429, 433, 435, 439, 443, 445, 449, 453, 457, 461},
					// Y
					{229, 233, 347, 353, 357, 363,
					 367, 373, 377, 383,
					 385, 389, 393, 397, 401, 407, 409, 413, 415, 419, 423, 427,
					 431, 437, 441, 447, 451, 455, 459, 463}});

			m_patch_facedir_next_cell_uid.push_back(
				{ // X
					#ifdef CORRECT_UID
					{ 99,  98,  -1, 101, 100,  -1,
					 103, 102,  -1, 105, 104,  -1,
					 107, 106, 110, 109, 108, 112, 111,  -1, 113,  -1,
					 115, 114, 118, 117, 116, 120, 119,  -1, 121,  -1},
					#else
					{ 99,  98,  34, 101, 100,  -1,  // error 34 -> -1
					 103, 102,  44, 105, 104,  -1,  // error 44 -> -1
					 107, 106, 110, 109, 108, 112, 111,  74, 113,  -1,  // error 74 -> -1
					 115, 114, 118, 117, 116, 120, 119,  76, 121,  -1},  // error 76 -> -1
					#endif
					// Y
					{ 98,  99, 100, 101, 102, 103,
					 104, 105,  -1,  -1,
					 106, 108, 107, 109, 114, 115, 110, 112, 111, 113, 118, 119,
					#ifdef CORRECT_UID
					 116, 117,  -1,  -1, 120, 121,  -1,  -1}});
					#else
					 116, 117, 126, 127, 120, 121,  99,  99}});  // error 126, 127 -> 98 or -1 et 99 -> -1
					#endif

			m_patch_facedir_prev_cell_uid.push_back(
				{ // X
					#ifdef CORRECT_UID
					{ 98,  -1,  99, 100,  -1, 101,
					 102,  -1, 103, 104,  -1, 105,
					 106,  -1, 107, 108,  -1, 109, 110, 111, 112, 113,
					 114,  -1, 115, 116,  -1, 117, 118, 119, 120, 121},
					#else
					{ 98, 123,  99, 100, 125, 101,  // error 123, 125 -> -1
					 102,  42, 103, 104,  -1, 105,  // error 42 -> -1
					 106,  67, 107, 108,  -1, 109, 110, 111, 112, 113,  // error 67 -> -1
					 114,  69, 115, 116,  -1, 117, 118, 119, 120, 121},  // error 69 -> -1
					#endif
					// Y
					#ifdef CORRECT_UID
					{ -1,  -1,  98,  99, 100, 101,
					 102, 103, 104, 105,
					  -1, 106,  -1, 107, 108, 109,  -1, 110,  -1, 111, 112, 113,
					 114, 115, 116, 117, 118, 119, 120, 121}});
					#else
					{ 72,  73,  98,  99, 100, 101,  // error 72, 73 -> -1
					 102, 103, 104, 105,
					  56, 106,  56, 107, 108, 109,  57, 110,  57, 111, 112, 113,  // error 56, 57 -> -1
					 114, 115, 116, 117, 118, 119, 120, 121}});
					#endif
		}

		// Patch 4 :
		{
			//  ---------492-----------------498---------
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 494       128       490       129       496
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  ---------482-----------------488---------
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 484       126       480       127       486
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  ---------441-----------------447---------

			m_patch_facedir_uid.push_back(
				{	// X
        	{480, 484, 486, 490, 494, 496},
					// Y
        	{441, 447, 482, 488, 492, 498}});

			m_patch_facedir_next_cell_uid.push_back(
				{ // X
					#ifdef CORRECT_UID
        	{127, 126,  -1, 129, 128,  -1},
					#else
        	{127, 126,  99, 129, 128,  -1},  // error 99 -> -1
					#endif
					// Y
					#ifdef CORRECT_UID
        	{126, 127, 128, 129,  -1,  -1}});
					#else
        	{126, 127, 128, 129, 100, 100}});  // error 100 -> -1
					#endif

			m_patch_facedir_prev_cell_uid.push_back(
				{ // X
					#ifdef CORRECT_UID
        	{126,  -1, 127, 128,  -1, 129},
					#else
        	{126, 123, 127, 128,  -1, 129},  // error 123 -> -1
					#endif
					// Y
					#ifdef CORRECT_UID
        	{ -1,  -1, 126, 127, 128, 129}});
					#else
        	{116, 117, 126, 127, 128, 129}});  // error 116, 117 -> -1
					#endif
		}
  }

	// Tableaux des vues cartésiennes sur les noeuds par patch
	{
		// Patch 0
		{

      //  55--------56--------57--------58--------59--------60--------61--------62--------63--------64--------65
      //  |         |         |         |         |         |         |         |         |         |         |
      //  |         |         |         |         |         |         |         |         |         |         |
      //  |         |         |         |         |         |         |         |         |         |         |
      //  44--------45--------46--------47--------48--------49--------50--------51--------52--------53--------54
      //  |         |         |         |         |         |         |         |         |         |         |
      //  |         |         |         |         |         |         |         |         |         |         |
      //  |         |         |         |         |         |         |         |         |         |         |
      //  33--------34--------35--------36--------37--------38--------39--------40--------41--------42--------43
      //  |         |         |         |         |         |         |         |         |         |         |
      //  |         |         |         |         |         |         |         |         |         |         |
      //  |         |         |         |         |         |         |         |         |         |         |
      //  22--------23--------24--------25--------26--------27--------28--------29--------30--------31--------32
      //  |         |         |         |         |         |         |         |         |         |         |
      //  |         |         |         |         |         |         |         |         |         |         |
      //  |         |         |         |         |         |         |         |         |         |         |
      //  11--------12--------13--------14--------15--------16--------17--------18--------19--------20--------21
      //  |         |         |         |         |         |         |         |         |         |         |
      //  |         |         |         |         |         |         |         |         |         |         |
      //  |         |         |         |         |         |         |         |         |         |         |
      //  0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------10

			m_patch_node_uid.push_back(
				{  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,
					11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
					22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,
					33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
					44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,
					55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65});

			m_patch_nodedir_next_uid.push_back(
				{	// X
					{  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  -1,
						12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  -1,
						23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  -1,
						34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  -1,
						45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  -1,
						56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  -1},
					// Y
					{ 11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
						22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,
						33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
						44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,
						55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
						-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1}});

			m_patch_nodedir_prev_uid.push_back(
				{	// X
					{ -1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
						-1,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,
						-1,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
						-1,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,
						-1,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,
						-1,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64},
					// Y
					{ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
						0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,
						11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
						22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,
						33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
						44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54}});
		}

		// Patch 1
		{
			//  35-------106--------36-------112--------37-------118--------38-------124--------39
			//  |         |         |         |         |         |         |         |         |
			//  |         |         |         |         |         |         |         |         |
			//  |         |         |         |         |         |         |         |         |
			// 102-------100-------104-------108-------110-------114-------116-------120-------122
			//  |         |         |         |         |         |         |         |         |
			//  |         |         |         |         |         |         |         |         |
			//  |         |         |         |         |         |         |         |         |
			//  24--------74--------25--------82--------26--------90--------27--------98--------28
			//  |         |         |         |         |         |         |         |         |
			//  |         |         |         |         |         |         |         |         |
			//  |         |         |         |         |         |         |         |         |
			//  70--------68--------72--------78--------80--------86--------88--------94--------96
			//  |         |         |         |         |         |         |         |         |
			//  |         |         |         |         |         |         |         |         |
			//  |         |         |         |         |         |         |         |         |
			//  13--------66--------14--------76--------15--------84--------16--------92--------17

			m_patch_node_uid.push_back(
				{ 13,  14,  15,  16,  17,
				  24,  25,  26,  27,  28,
					35,  36,  37,  38,  39,
					66,  68,  70,  72,  74,  76,  78,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98,
				 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124});

			m_patch_nodedir_next_uid.push_back(
				{	// X
					{ 66,  76,  84,  92,  -1,
						74,  82,  90,  98,  -1,
					 106, 112, 118, 124,  -1,
						14,  72,  68,  78,  25,  15,  80,  86,  26,  16,  88,  94,  27,  17,  96,  -1,  28,
					 104, 100, 108,  36, 110, 114,  37, 116, 120,  38, 122,  -1,  39},
					// Y
					{ 70,  72,  80,  88,  96,
					 102, 104, 110, 116, 122,
					  -1,  -1,  -1,  -1,  -1,
						68,  74,  24,  25, 100,  78,  82,  26, 108,  86,  90,  27, 114,  94,  98,  28, 120,
					 106,  35,  36,  -1, 112,  37,  -1, 118,  38,  -1, 124,  39,  -1}});

			m_patch_nodedir_prev_uid.push_back(
				{	// X
					{ -1,  66,  76,  84,  92,
						-1,  74,  82,  90,  98,
						-1, 106, 112, 118, 124,
						13,  70,  -1,  68,  24,  14,  72,  78,  25,  15,  80,  86,  26,  16,  88,  94,  27,
					 102,  -1, 100,  35, 104, 108,  36, 110, 114,  37, 116, 120,  38},
					// Y
					{ -1,  -1,  -1,  -1,  -1,
						70,  72,  80,  88,  96,
					 102, 104, 110, 116, 122,
						-1,  66,  13,  14,  68,  -1,  76,  15,  78,  -1,  84,  16,  86,  -1,  92,  17,  94,
					  74,  24,  25, 100,  82,  26, 108,  90,  27, 114,  98,  28, 120}});
		}

		// Patch 2 :
		{
			//  61-------149--------62-------155--------63
			//  |         |         |         |         |
			//  |         |         |         |         |
			//  |         |         |         |         |
			// 145-------143-------147-------151-------153
			//  |         |         |         |         |
			//  |         |         |         |         |
			//  |         |         |         |         |
			//  50-------133--------51-------141--------52
			//  |         |         |         |         |
			//  |         |         |         |         |
			//  |         |         |         |         |
			// 129-------127-------131-------137-------139
			//  |         |         |         |         |
			//  |         |         |         |         |
			//  |         |         |         |         |
			//  39-------125--------40-------135--------41

			m_patch_node_uid.push_back(
				{ 39,  40,  41,
				  50,  51,  52,
					61,  62,  63,
				 125, 127, 129, 131, 133, 135, 137, 139, 141,
				 143, 145, 147, 149, 151, 153, 155});

			m_patch_nodedir_next_uid.push_back(
				{	// X
					{125, 135,  -1,
				   133, 141,  -1,
				   149, 155,  -1,
				    40, 131, 127, 137,  51,  41, 139,  -1,  52,
				   147, 143, 151,  62, 153,  -1,  63},
					// Y
					{129, 131, 139,
				   145, 147, 153,
				    -1,  -1,  -1,
				   127, 133,  50,  51, 143, 137, 141,  52, 151,
				   149,  61,  62,  -1, 155,  63,  -1}});

			m_patch_nodedir_prev_uid.push_back(
				{	// X
					{ -1, 125, 135,
				  	-1, 133, 141,
				  	-1, 149, 155,
				    39, 129,  -1, 127,  50,  40, 131, 137,  51,
				   145,  -1, 143,  61, 147, 151,  62},
					// Y
					{ -1,  -1,  -1,
				   129, 131, 139,
				   145, 147, 153,
				    -1, 125,  39,  40, 127,  -1, 135,  41, 137,
				   133,  50,  51, 143, 141,  52, 151}});
		}

		// Patch 3 :
		{
			//  58-----------------170------------------59
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 166-----------------164-----------------168
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  47-----------------162------------------48
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 158-----------------156-----------------160
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  36-----------------112------------------37
			//  36-------196-------112-------202--------37
			//  |         |         |         |         |
			//  |         |         |         |         |
			//  |         |         |         |         |
			// 192-------190-------194-------198-------200
			//  |         |         |         |         |
			//  |         |         |         |         |
			//  |         |         |         |         |
			// 104-------180-------108-------188-------110
			//  |         |         |         |         |
			//  |         |         |         |         |
			//  |         |         |         |         |
			// 176-------174-------178-------184-------186
			//  |         |         |         |         |
			//  |         |         |         |         |
			//  |         |         |         |         |
			//  25-------172--------82-------182--------26

			m_patch_node_uid.push_back(
				{ 25,  26,  36,  37,  47,  48,  58,  59,
				  82, 104, 108, 110, 112,
				 156, 158, 160, 162, 164, 166, 168, 170,
				 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202});

			m_patch_nodedir_next_uid.push_back(
				{	// X
					{172,  -1, 196,  -1, 162,  -1, 170,  -1,
					 182, 180, 188,  -1, 202,
					 160, 156,  -1,  48, 168, 164,  -1,  59,
					  82, 178, 174, 184, 108,  26, 186,  -1, 110, 194, 190, 198, 112, 200,  -1,  37},
					// Y
					{176, 186, 158, 160, 166, 168,  -1,  -1,
					 178, 192, 194, 200, 156,
					 162,  47,  48, 164, 170,  58,  59,  -1,
					 174, 180, 104, 108, 190, 184, 188, 110, 198, 196,  36, 112,  -1, 202,  37,  -1}});

			m_patch_nodedir_prev_uid.push_back(
				{	// X
					{ -1, 182,  -1, 202,  -1, 162,  -1, 170,
					 172,  -1, 180, 188, 196,
					 158,  -1, 156,  47, 166,  -1, 164,  58,
					  25, 176,  -1, 174, 104,  82, 178, 184, 108, 192,  -1, 190,  36, 194, 198, 112},
					// Y
					{ -1,  -1, 192, 200, 158, 160, 166, 168,
					  -1, 176, 178, 186, 194,
					 112,  36,  37, 156, 162,  47,  48, 164,
					  -1, 172,  25,  82, 174,  -1, 182,  26, 184, 180, 104, 108, 190, 188, 110, 198}});
		}

		// Patch 4 :
		{
			// 158-----------------215-----------------156
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 211-----------------209-----------------213
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  36-----------------196-----------------112

			m_patch_node_uid.push_back(
				{ 36, 112, 156, 158, 196, 209, 211, 213, 215});

			m_patch_nodedir_next_uid.push_back(
				{	// X
					{196,  -1,  -1, 215, 112, 213, 209,  -1, 156},
					// Y
					{211, 213,  -1,  -1, 209, 215, 158, 156,  -1}});

			m_patch_nodedir_prev_uid.push_back(
				{	// X
					{ -1, 196, 215,  -1,  36, 211,  -1, 209, 158},
					// Y
					{ -1,  -1, 213, 211,  -1, 196,  36, 112, 209}});
		}
	}

	// Tableaux des connectivités noeuds -> mailles par patch
	{
		// Patch -1
		{
			//  55--------56--------57--------58xxxxxxxxxx59--------60--------61oooooooo62oooooooo63--------64--------65
			//  |         |         |         x     |     x         |         o    |    |    |    o         |         |
			//  |         |         |         x 104 | 105 x         |         o 92 | 93 | 96 | 97 o         |         |
			//  |         |         |         x     |     x         |         o    |    |    |    o         |         |
			//  |    40   |    41   |    42   x-----43----x    44   |    45   o----46---|----47---o    48   |    49   |
			//  |         |         |         x     |     x         |         o    |    |    |    o         |         |
			//  |         |         |         x 102 | 103 x         |         o 90 | 91 | 94 | 95 o         |         |
			//  |         |         |         x     |     x         |         o    |    |    |    o         |         |
			//  44--------45--------46--------47----------48--------49--------50---|----51---|----52--------53--------54
			//  |         |         |         x     |     x         |         o    |    |    |    o         |         |
			//  |         |         |         x 100 | 101 x         |         o 84 | 85 | 88 | 89 o         |         |
			//  |         |         |         x     |     x         |         o    |    |    |    o         |         |
			//  |    30   |    31   |    32   ======33----x    34   |    35   o----36---|----37---o    38   |    39   |
			//  |         |         |         =28|29=     x         |         o    |    |    |    o         |         |
			//  |         |         |         =--98-=  99 x         |         o 82 | 83 | 86 | 87 o         |         |
			//  |         |         |         =26|27=     x         |         o    |    |    |    o         |         |
			//  33--------34--------35++++++++36=====+++++37++++++++38++++++++39oooooooo40oooooooo41--------42--------43
			//  |         |         +    |    x16|17|20|21x    |    |    |    +         |         |         |         |
			//  |         |         + 68 | 69 x--72-|--73-x 76 | 77 | 80 | 81 +         |         |         |         |
			//  |         |         +    |    x14|15|18|19x    |    |    |    +         |         |         |         |
			//  |    20   |    21   +----22---x-----23----x----24---|----25---+    26   |    27   |    28   |    29   |
			//  |         |         +    |    x08|09|12|13x    |    |    |    +         |         |         |         |
			//  |         |         + 66 | 67 x--70-|--71-x 74 | 75 | 78 | 79 +         |         |         |         |
			//  |         |         +    |    x06|07|10|11x    |    |    |    +         |         |         |         |
			//  22--------23--------24--------25xxxxxxxxxx26--------27--------28--------29--------30--------31--------32
			//  |         |         +    |    |     |     |    |    |    |    +         |         |         |         |
			//  |         |         + 52 | 53 |  56 |  57 | 60 | 61 | 64 | 65 +         |         |         |         |
			//  |         |         +    |    |     |     |    |    |    |    +         |         |         |         |
			//  |    10   |    11   +----12---|-----13----|----14---|----15---+    16   |    17   |    18   |    19   |
			//  |         |         +    |    |     |     |    |    |    |    +         |         |         |         |
			//  |         |         + 50 | 51 |  54 |  55 | 58 | 59 | 62 | 63 +         |         |         |         |
			//  |         |         +    |    |     |     |    |    |    |    +         |         |         |         |
			//  11--------12--------13++++++++14++++++++++15++++++++16++++++++17--------18--------19--------20--------21
			//  |         |         |         |           |         |         |         |         |         |         |
			//  |         |         |         |           |         |         |         |         |         |         |
			//  |         |         |         |           |         |         |         |         |         |         |
			//  |    0    |    1    |    2    |     3     |    4    |    5    |     6   |    7    |    8    |    9    |
			//  |         |         |         |           |         |         |         |         |         |         |
			//  |         |         |         |           |         |         |         |         |         |         |
			//  |         |         |         |           |         |         |         |         |         |         |
			//  0---------1---------2---------3-----------4---------5---------6---------7---------8---------9---------10

			m_patch_nodecell_upper_right_uid.push_back(
				{  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  -1,
					10,  11,  50,  54,  58,  62,  16,  17,  18,  19,  -1,
					20,  21,  66, 106,  74,  78,  26,  27,  28,  29,  -1,
					30,  31, 122, 126,  34,  35,  82,  86,  38,  39,  -1,
					40,  41,  42, 102,  44,  45,  90,  94,  48,  49,  -1,
					-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1});

			m_patch_nodecell_upper_left_uid.push_back(
				{ -1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
					-1,  10,  11,  51,  55,  59,  63,  16,  17,  18,  19,
					-1,  20,  21,  67, 111,  75,  79,  26,  27,  28,  29,
					-1,  30,  31, 123,  99,  34,  35,  83,  87,  38,  39,
					-1,  40,  41,  42, 103,  44,  45,  91,  95,  48,  49,
					-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1});

			m_patch_nodecell_lower_right_uid.push_back(
				{ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
						0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  -1,
					10,  11,  52,  56,  60,  64,  16,  17,  18,  19,  -1,
					20,  21,  68, 116,  76,  80,  26,  27,  28,  29,  -1,
					30,  31, 124, 100,  34,  35,  84,  88,  38,  39,  -1,
					40,  41,  42, 104,  44,  45,  92,  96,  48,  49,  -1});

			m_patch_nodecell_lower_left_uid.push_back(
				{ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
					-1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
					-1,  10,  11,  53,  57,  61,  65,  16,  17,  18,  19,
					-1,  20,  21,  69, 121,  77,  81,  26,  27,  28,  29,
					-1,  30,  31, 125, 101,  34,  35,  85,  89,  38,  39,
					-1,  40,  41,  42, 105,  44,  45,  93,  97,  48,  49});
		}

		// Patch 0
		{
    	//  55--------56--------57--------58--------59--------60--------61--------62--------63--------64--------65
    	//  |         |         |         |         |         |         |         |         |         |         |
    	//  |    40   |    41   |    42   |    43   |    44   |    45   |    46   |    47   |    48   |    49   |
    	//  |         |         |         |         |         |         |         |         |         |         |
    	//  44--------45--------46--------47--------48--------49--------50--------51--------52--------53--------54
    	//  |         |         |         |         |         |         |         |         |         |         |
    	//  |    30   |    31   |    32   |    33   |    34   |    35   |    36   |    37   |    38   |    39   |
    	//  |         |         |         |         |         |         |         |         |         |         |
    	//  33--------34--------35--------36--------37--------38--------39--------40--------41--------42--------43
    	//  |         |         |         |         |         |         |         |         |         |         |
    	//  |    20   |    21   |    22   |    23   |    24   |    25   |    26   |    27   |    28   |    29   |
    	//  |         |         |         |         |         |         |         |         |         |         |
    	//  22--------23--------24--------25--------26--------27--------28--------29--------30--------31--------32
    	//  |         |         |         |         |         |         |         |         |         |         |
    	//  |    10   |    11   |    12   |    13   |    14   |    15   |    16   |    17   |    18   |    19   |
    	//  |         |         |         |         |         |         |         |         |         |         |
    	//  11--------12--------13--------14--------15--------16--------17--------18--------19--------20--------21
    	//  |         |         |         |         |         |         |         |         |         |         |
    	//  |    0    |    1    |    2    |    3    |    4    |    5    |     6   |    7    |    8    |    9    |
    	//  |         |         |         |         |         |         |         |         |         |         |
    	//  0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------10

			m_patch_nodecell_upper_right_uid.push_back(
				{  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  -1,
					10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  -1,
					20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  -1,
					30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  -1,
					40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  -1,
					-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1});

			m_patch_nodecell_upper_left_uid.push_back(
				{ -1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
					-1,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
					-1,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
					-1,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
					-1,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
					-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1});

			m_patch_nodecell_lower_right_uid.push_back(
				{ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
					0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  -1,
					10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  -1,
					20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  -1,
					30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  -1,
					40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  -1});

			m_patch_nodecell_lower_left_uid.push_back(
				{ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
					-1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
					-1,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
					-1,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
					-1,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
					-1,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49});
		}

		// Patch 1
		{
			//  35-------106--------36-------112--------37-------118--------38-------124--------39
			//  |         |         |         |         |         |         |         |         |
			//  |    68   |    69   |    72   |    73   |    76   |    77   |    80   |    81   |
			//  |         |         |         |         |         |         |         |         |
			// 102-------100-------104-------108-------110-------114-------116-------120-------122
			//  |         |         |         |         |         |         |         |         |
			//  |    66   |    67   |    70   |    71   |    74   |    75   |    78   |    79   |
			//  |         |         |         |         |         |         |         |         |
			//  24--------74--------25--------82--------26--------90--------27--------98--------28
			//  |         |         |         |         |         |         |         |         |
			//  |    52   |    53   |    56   |    57   |    60   |    61   |    64   |    65   |
			//  |         |         |         |         |         |         |         |         |
			//  70--------68--------72--------78--------80--------86--------88--------94--------96
			//  |         |         |         |         |         |         |         |         |
			//  |    50   |    51   |    54   |    55   |    58   |    59   |    62   |    63   |
			//  |         |         |         |         |         |         |         |         |
			//  13--------66--------14--------76--------15--------84--------16--------92--------17

			m_patch_nodecell_upper_right_uid.push_back(
				{ 50,  54,  58,  62,  -1,
				  66,  70,  74,  78,  -1,
					#ifdef CORRECT_UID
				  -1,  -1,  -1,  -1,  -1,
					#else
				 122,  98,  -1,  -1,  82,  // error 122, 98, 82 -> -1
					#endif
					51,  53,  52,  56,  67,  55,  57,  60,  71,  59,  61,  64,  75,  63,  65,  -1,  79,
					#ifdef CORRECT_UID
				  69,  68,  72,  -1,  73,  76,  -1,  77,  80,  -1,  81,  -1,  -1});
					#else
				  69,  68,  72, 123,  73,  76,  99,  77,  80,  -1,  81,  -1,  -1});  // error 123, 99 -> -1
					#endif

			m_patch_nodecell_upper_left_uid.push_back(
				{ -1,  51,  55,  59,  63,
				  -1,  67,  71,  75,  79,
					#ifdef CORRECT_UID
					-1,  -1,  -1,  -1,  -1,
					#else
					-1, 123,  99,  -1,  -1,  // error 123, 99 -> -1
					#endif
					50,  52,  -1,  53,  66,  54,  56,  57,  70,  58,  60,  61,  74,  62,  64,  65,  78,
					#ifdef CORRECT_UID
				  68,  -1,  69,  -1,  72,  73,  -1,  76,  77,  -1,  80,  81,  -1});
					#else
				  68,  -1,  69, 122,  72,  73,  98,  76,  77,  -1,  80,  81,  -1});  // error 122, 98 -> -1
					#endif

			m_patch_nodecell_lower_right_uid.push_back(
				{ -1,  -1,  -1,  -1,  -1,
				  52,  56,  60,  64,  -1,
					68,  72,  76,  80,  -1,
					-1,  51,  50,  54,  53,  -1,  55,  58,  57,  -1,  59,  62,  61,  -1,  63,  -1,  65,
				  67,  66,  70,  69,  71,  74,  73,  75,  78,  77,  79,  -1,  81});

			m_patch_nodecell_lower_left_uid.push_back(
				{ -1,  -1,  -1,  -1,  -1,
				  -1,  53,  57,  61,  65,
					-1,  69,  73,  77,  81,
					-1,  50,  -1,  51,  52,  -1,  54,  55,  56,  -1,  58,  59,  60,  -1,  62,  63,  64,
				  66,  -1,  67,  68,  70,  71,  72,  74,  75,  76,  78,  79,  80});
		}

		// Patch 2 :
		{
			//  61-------149--------62-------155--------63
			//  |         |         |         |         |
			//  |    92   |    93   |    96   |    97   |
			//  |         |         |         |         |
			// 145-------143-------147-------151-------153
			//  |         |         |         |         |
			//  |    90   |    91   |    94   |    95   |
			//  |         |         |         |         |
			//  50-------133--------51-------141--------52
			//  |         |         |         |         |
			//  |    84   |    85   |    88   |    89   |
			//  |         |         |         |         |
			// 129-------127-------131-------137-------139
			//  |         |         |         |         |
			//  |    82   |    83   |    86   |    87   |
			//  |         |         |         |         |
			//  39-------125--------40-------135--------41

			m_patch_nodecell_upper_right_uid.push_back(
				{ 82,  86,  -1,
				  90,  94,  -1,
					-1,  -1,  -1,
				  83,  85,  84,  88,  91,  87,  89,  -1,  95,
				  93,  92,  96,  -1,  97,  -1,  -1});

			m_patch_nodecell_upper_left_uid.push_back(
				{ -1,  83,  87,
				  -1,  91,  95,
					-1,  -1,  -1,
				  82,  84,  -1,  85,  90,  86,  88,  89,  94,
				  92,  -1,  93,  -1,  96,  97,  -1});

			m_patch_nodecell_lower_right_uid.push_back(
				{ -1,  -1,  -1,
				  84,  88,  -1,
					92,  96,  -1,
				  -1,  83,  82,  86,  85,  -1,  87,  -1,  89,
				  91,  90,  94,  93,  95,  -1,  97});

			m_patch_nodecell_lower_left_uid.push_back(
				#ifdef CORRECT_UID
				{ -1,  -1,  -1,
				#else
				{ 81,  -1,  -1,  // error 81 -> -1
				#endif
				  -1,  85,  89,
					-1,  93,  97,
				  -1,  82,  -1,  83,  84,  -1,  86,  87,  88,
				  90,  -1,  91,  92,  94,  95,  96});
		}

		// Patch 3 :
		{
			//  58-----------------170------------------59
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |        104        |        105        |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 166-----------------164-----------------168
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |        102        |        103        |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  47-----------------162------------------48
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |        100        |        101        |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 158<------347------>156<------353------>160
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |         98        |         99        |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  36-----------------112------------------37
			//  36-------196-------112-------202--------37
			//  |         |         |         |         |
			//  |    116  |   117   |   120   |   121   |
			//  |         |         |         |         |
			// 192-------190-------194-------198-------200
			//  |         |         |         |         |
			//  |   114   |   115   |   118   |   119   |
			//  |         |         |         |         |
			// 104-------180-------108-------188-------110
			//  |         |         |         |         |
			//  |   108   |   109   |   112   |   113   |
			//  |         |         |         |         |
			// 176-------174-------178-------184-------186
			//  |         |         |         |         |
			//  |   106   |   107   |   110   |   111   |
			//  |         |         |         |         |
			//  25-------172--------82-------182--------26

			m_patch_nodecell_upper_right_uid.push_back(
				#ifdef CORRECT_UID
				{106,  -1,  98,  -1, 102,  -1,  -1,  -1,
				 110, 114, 118,  -1,  99,
				 101, 100,  -1, 103, 105, 104,  -1,  -1,
				 107, 109, 108, 112, 115, 111, 113,  -1, 119, 117, 116, 120,  -1, 121,  -1,  -1});
				#else
				{106,  -1, 126,  -1,  -1,  -1,  -1,  -1,  // error 126 -> 98, -1 -> 102
				 110, 114, 118,  -1,  -1,  // error -1 -> 99
				  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  // error -1 -> 101, 100, 103, 105, 104
				 107, 109, 108, 112, 115, 111, 113,  -1, 119, 117, 116, 120, 127, 121,  -1,  -1});  // error 127 -> -1
				#endif

			m_patch_nodecell_upper_left_uid.push_back(
				#ifdef CORRECT_UID
				{ -1, 111,  -1,  99,  -1, 103,  -1,  -1,
				 107,  -1, 115, 119,  98,
				 100,  -1, 101, 102, 104,  -1, 105,  -1,
				 106, 108,  -1, 109, 114, 110, 112, 113, 118, 116,  -1, 117,  -1, 120, 121,  -1});
				#else
				{ -1, 111,  -1,  -1,  -1,  -1,  -1,  -1,  // error -1 -> 99, 103
				 107,  -1, 115, 119, 127,  // error 127 -> 98
				  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  // error -1 -> 100, 101, 102, 104, 105
				 106, 108,  -1, 109, 114, 110, 112, 113, 118, 116,  -1, 117, 126, 120, 121,  -1});  // error 126 -> -1
				#endif

			m_patch_nodecell_lower_right_uid.push_back(
				#ifdef CORRECT_UID
				{ -1,  -1, 116,  -1, 100,  -1, 104,  -1,
				  -1, 108, 112,  -1, 120,
				  99,  98,  -1, 101, 103, 102,  -1, 105,
				  -1, 107, 106, 110, 109,  -1, 111,  -1, 113, 115, 114, 118, 117, 119,  -1, 121});
				#else
				{ -1,  -1, 116,  -1,  -1,  -1,  -1,  -1,  // error -1 -> 100, 104
				  -1, 108, 112,  -1, 120,
				  -1, 128,  -1,  -1,  -1,  -1,  -1,  -1,  // error 128 -> 98, -1 -> 99, 101, 103, 102, 105
				  -1, 107, 106, 110, 109,  -1, 111,  -1, 113, 115, 114, 118, 117, 119,  -1, 121});
				#endif

			m_patch_nodecell_lower_left_uid.push_back(
				#ifdef CORRECT_UID
				{ -1,  -1,  -1, 121,  -1, 101,  -1, 105,
				  -1,  -1, 109, 113, 117,
				  98,  -1,  99, 100, 102,  -1, 103, 104,
				  -1, 106,  -1, 107, 108,  -1, 110, 111, 112, 114,  -1, 115, 116, 118, 119, 120});
				#else
				{ -1,  -1,  -1, 121,  -1,  -1,  -1,  -1, // error -1 -> 101, 105
				  -1,  -1, 109, 113, 117,
				 129,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  // error 129 -> 98, -1 -> 99, 100, 102, 103, 104
				  -1, 106,  -1, 107, 108,  -1, 110, 111, 112, 114,  -1, 115, 116, 118, 119, 120});
				#endif
		}

		// Patch 4 :
		{
			// 158-----------------215-----------------156
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |        128        |        129        |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			// 211-----------------209-----------------213
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  |        126        |        127        |
			//  |                   |                   |
			//  |                   |                   |
			//  |                   |                   |
			//  36-----------------196-----------------112

			m_patch_node_uid.push_back(
				{ 36, 112, 156, 158, 196, 209, 211, 213, 215});

			m_patch_nodecell_upper_right_uid.push_back(
				{126,  -1,  -1,  -1, 127, 129, 128,  -1,  -1});

			m_patch_nodecell_upper_left_uid.push_back(
				{ -1, 127,  -1,  -1, 126, 128,  -1, 129,  -1});

			m_patch_nodecell_lower_right_uid.push_back(
				#ifdef CORRECT_UID
				{ -1,  -1,  -1, 128,  -1, 127, 126,  -1, 129});
				#else
				{116, 120,  -1, 128, 117, 127, 126,  -1, 129});  // error 116, 120, 117 -> -1
				#endif

			m_patch_nodecell_lower_left_uid.push_back(
				#ifdef CORRECT_UID
				{ -1,  -1, 129,  -1,  -1, 126,  -1, 127, 128});
				#else
				{ -1, 117, 129,  -1, 116, 126,  -1, 127, 128});  // error 117, 116 -> -1
				#endif
		}
	}
// clang-format on
}

/*---------------------------------------------------------------------------*/
/*!
 * \brief
 * Actions a effectuer avant chaque test
 */
/*---------------------------------------------------------------------------*/
void
UnitTestCartesianMeshPatchService::setUp()
{
}

/*---------------------------------------------------------------------------*/
/*!
 * \brief
 * Actions a effectuer apres chaque test
 */
/*---------------------------------------------------------------------------*/
void
UnitTestCartesianMeshPatchService::tearDown()
{
}

/*---------------------------------------------------------------------------*/
/*!
 * \brief
 * Actions a effectuer apres tous les tests
 */
/*---------------------------------------------------------------------------*/
void
UnitTestCartesianMeshPatchService::tearDownForClass()
{
}

/*---------------------------------------------------------------------------*/
/*!
 * \brief
 * Test sur les mailles par niveau de raffinement et leurs parents
 */
/*---------------------------------------------------------------------------*/
void
UnitTestCartesianMeshPatchService::testCartesianMeshPatchCellsAndParents()
{
  for (Int32 lvl{0}; lvl < 3; ++lvl) {
    const std::vector<Int64>& cell_uid{m_lvl_cell_uid[lvl]};
    const std::vector<Int64>& cell_p_uid{m_lvl_cell_p_uid[lvl]};
    const std::vector<Int64>& cell_tp_uid{m_lvl_cell_tp_uid[lvl]};

    const CellGroup& allLevelCells{this->mesh()->allLevelCells(lvl)};
    ASSERT_EQUAL(static_cast<Integer>(cell_uid.size()), allLevelCells.size());

    ENUMERATE_CELL (cell_i, allLevelCells) {
      const Int64 i{cell_i.index()};

      ASSERT_EQUAL(cell_uid[i], cell_i->uniqueId().asInt64());
      if (lvl > 0) {
        ASSERT_EQUAL(cell_p_uid[i], cell_i->hParent().uniqueId().asInt64());
        ASSERT_EQUAL(cell_tp_uid[i], cell_i->topHParent().uniqueId().asInt64());
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*!
 * \brief
 * Test vue cartésienne sur les mailles
 */
/*---------------------------------------------------------------------------*/
void
UnitTestCartesianMeshPatchService::testCartesianMeshPatchCellDirectionMng()
{
  ASSERT_EQUAL(m_cartesian_mesh->nbPatch(), 5);

  for (Int32 patch_i{-1}; patch_i < 5; ++patch_i) {
    ICartesianMeshPatch* cm_patch{(patch_i == -1) ? nullptr : m_cartesian_mesh->patch(patch_i)};

    const Int32 ipatch{(patch_i == -1) ? 0 : patch_i};
    const std::vector<Int64>& cell_uid{m_patch_cell_uid[ipatch]};
    const std::vector<std::vector<Int64>>& celldir_next_uid{m_patch_celldir_next_uid[ipatch]};
    const std::vector<std::vector<Int64>>& celldir_prev_uid{m_patch_celldir_prev_uid[ipatch]};
    const std::vector<std::vector<Int64>>& cellfacedir_next_uid{m_patch_cellfacedir_next_uid[ipatch]};
    const std::vector<std::vector<Int64>>& cellfacedir_prev_uid{m_patch_cellfacedir_prev_uid[ipatch]};
    const std::vector<Int64>& cellnode_upper_right_uid{m_patch_cellnode_upper_right_uid[ipatch]};
    const std::vector<Int64>& cellnode_upper_left_uid{m_patch_cellnode_upper_left_uid[ipatch]};
    const std::vector<Int64>& cellnode_lower_right_uid{m_patch_cellnode_lower_right_uid[ipatch]};
    const std::vector<Int64>& cellnode_lower_left_uid{m_patch_cellnode_lower_left_uid[ipatch]};

    for (Int32 dir{0}; dir < 2; ++dir) {
      CellDirectionMng cell_dm{(patch_i == -1) ? m_cartesian_mesh->cellDirection(dir) : cm_patch->cellDirection(dir)};

      info() << "Patch = " << patch_i << " dir=" << dir;
      const std::vector<Int64>& cell_next_uid{celldir_next_uid[dir]};
      const std::vector<Int64>& cell_prev_uid{celldir_prev_uid[dir]};
      const std::vector<Int64>& cellface_next_uid{cellfacedir_next_uid[dir]};
      const std::vector<Int64>& cellface_prev_uid{cellfacedir_prev_uid[dir]};

      const CellGroup& allCellDMCells{cell_dm.allCells()};
      ASSERT_EQUAL(static_cast<Integer>(cell_uid.size()), allCellDMCells.size());

      ENUMERATE_CELL (cell_i, allCellDMCells) {
        const Int64 i{cell_i.index()};
        const Cell& cell{*cell_i};

        ASSERT_EQUAL(cell_uid[i], cell_i->uniqueId().asInt64());

        {
          DirCell dir_cell{cell_dm[cell_i]};
          info() << "Cell = " << cell_i->uniqueId() << " expected_next=" << cell_next_uid[i] << " expected_prev=" << cell_prev_uid[i];
          ASSERT_EQUAL(cell_next_uid[i], dir_cell.next().uniqueId().asInt64());
          ASSERT_EQUAL(cell_prev_uid[i], dir_cell.previous().uniqueId().asInt64());
        }

        {
          const DirCellFace& dir_cellface{cell_dm.cellFace(cell)};
          ASSERT_EQUAL(cellface_next_uid[i], dir_cellface.next().uniqueId().asInt64());
          ASSERT_EQUAL(cellface_prev_uid[i], dir_cellface.previous().uniqueId().asInt64());
        }

        {
          const DirCellNode& dir_cellnode{cell_dm.cellNode(cell)};
          {
            const Node& node_upper_right{(dir == 1) ? dir_cellnode.nextRight() : dir_cellnode.nextLeft()};
            ASSERT_EQUAL(cellnode_upper_right_uid[i], node_upper_right.uniqueId().asInt64());
          }
          {
            const Node& node_upper_left{(dir == 1) ? dir_cellnode.nextLeft() : dir_cellnode.previousLeft()};
            ASSERT_EQUAL(cellnode_upper_left_uid[i], node_upper_left.uniqueId().asInt64());
          }
          {
            const Node& node_lower_right{(dir == 1) ? dir_cellnode.previousRight() : dir_cellnode.nextRight()};
            ASSERT_EQUAL(cellnode_lower_right_uid[i], node_lower_right.uniqueId().asInt64());
          }
          {
            const Node& node_lower_left{(dir == 1) ? dir_cellnode.previousLeft() : dir_cellnode.previousRight()};
            ASSERT_EQUAL(cellnode_lower_left_uid[i], node_lower_left.uniqueId().asInt64());
          }
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*!
 * \brief
 * Test vue cartésienne sur les faces
 */
/*---------------------------------------------------------------------------*/
void
UnitTestCartesianMeshPatchService::testCartesianMeshPatchFaceDirectionMng()
{
  for (Int32 patch_i{-1}; patch_i < 5; ++patch_i) {
    ICartesianMeshPatch* cm_patch{(patch_i == -1) ? nullptr : m_cartesian_mesh->patch(patch_i)};

    const Int32 ipatch{(patch_i == -1) ? 0 : patch_i};
    const std::vector<std::vector<Int64>>& facedir_uid{m_patch_facedir_uid[ipatch]};
    const std::vector<std::vector<Int64>>& facedir_next_cell_uid{m_patch_facedir_next_cell_uid[ipatch]};
    const std::vector<std::vector<Int64>>& facedir_prev_cell_uid{m_patch_facedir_prev_cell_uid[ipatch]};

    for (Int32 dir{0}; dir < 2; ++dir) {
      FaceDirectionMng face_dm{(patch_i == -1) ? m_cartesian_mesh->faceDirection(dir) : cm_patch->faceDirection(dir)};

      info() << "Patch = " << patch_i << " dir=" << dir;

      const std::vector<Int64>& face_uid{facedir_uid[dir]};
      const std::vector<Int64>& face_next_cell_uid{facedir_next_cell_uid[dir]};
      const std::vector<Int64>& face_prev_cell_uid{facedir_prev_cell_uid[dir]};

      const FaceGroup& allFaceDMFaces{face_dm.allFaces()};
      ASSERT_EQUAL(static_cast<Integer>(face_uid.size()), allFaceDMFaces.size());

      ENUMERATE_FACE (face_i, allFaceDMFaces) {
        const Int64 i{face_i.index()};

        DirFace dir_face{face_dm[face_i]};
        Cell next_cell = dir_face.nextCell();
        Cell prev_cell = dir_face.previousCell();
        ASSERT_EQUAL(face_prev_cell_uid[i], dir_face.previousCell().uniqueId().asInt64());
        info() << "Face = " << ItemPrinter(*face_i) << " next_cell=" << ItemPrinter(next_cell)
               << " prev_cell=" << ItemPrinter(prev_cell);
        info() << "Face = " << face_i->uniqueId() << " expected_next=" << face_next_cell_uid[i]
               << " expected_prev=" << face_prev_cell_uid[i];
        ASSERT_EQUAL(face_uid[i], face_i->uniqueId().asInt64());

        ASSERT_EQUAL(face_next_cell_uid[i], dir_face.nextCell().uniqueId().asInt64());
        ASSERT_EQUAL(face_prev_cell_uid[i], dir_face.previousCell().uniqueId().asInt64());
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*!
 * \brief
 * Test vue cartésienne sur les noeuds
 */
/*---------------------------------------------------------------------------*/
void
UnitTestCartesianMeshPatchService::testCartesianMeshPatchNodeDirectionMng()
{
  for (Int32 patch_i{-1}; patch_i < 5; ++patch_i) {
    ICartesianMeshPatch* cm_patch{(patch_i == -1) ? nullptr : m_cartesian_mesh->patch(patch_i)};

    const Int32 ipatch{(patch_i == -1) ? 0 : patch_i};
    const std::vector<Int64>& node_uid{m_patch_node_uid[ipatch]};
    const std::vector<std::vector<Int64>>& nodedir_next_uid{m_patch_nodedir_next_uid[ipatch]};
    const std::vector<std::vector<Int64>>& nodedir_prev_uid{m_patch_nodedir_prev_uid[ipatch]};

    for (Int32 dir{0}; dir < 2; ++dir) {
      NodeDirectionMng node_dm{(patch_i == -1) ? m_cartesian_mesh->nodeDirection(dir) : cm_patch->nodeDirection(dir)};

      const std::vector<Int64>& node_next_uid{nodedir_next_uid[dir]};
      const std::vector<Int64>& node_prev_uid{nodedir_prev_uid[dir]};

      const NodeGroup& allNodeDMNodes{node_dm.allNodes()};
      ASSERT_EQUAL(static_cast<Integer>(node_uid.size()), allNodeDMNodes.size());

      ENUMERATE_NODE (node_i, allNodeDMNodes) {
        const Int64 i{node_i.index()};

        ASSERT_EQUAL(node_uid[i], node_i->uniqueId().asInt64());

        const DirNode& dir_node{node_dm[node_i]};
        ASSERT_EQUAL(node_next_uid[i], dir_node.next().uniqueId().asInt64());
        ASSERT_EQUAL(node_prev_uid[i], dir_node.previous().uniqueId().asInt64());
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*!
 * \brief
 * Test connectivité cartésienne maille -> noeud et noeud -> maille
 */
/*---------------------------------------------------------------------------*/
void
UnitTestCartesianMeshPatchService::testCartesianMeshPatchCartesianConnectivity()
{
  // Ne fait pas pour l'instant car il ne fonctionne pas.
  warning() << A_FUNCINFO << " This test is not activated because it does not (yet) work.";
  return;

  const CartesianConnectivity& cc{m_cartesian_mesh->connectivity()};

  for (Int32 patch_i{-1}; patch_i < 5; ++patch_i) {
    ICartesianMeshPatch* cm_patch{(patch_i == -1) ? nullptr : m_cartesian_mesh->patch(patch_i)};

    {
      const Int32 ipatch{(patch_i == -1) ? 0 : patch_i};

      const std::vector<Int64>& cellnode_upper_right_uid{m_patch_cellnode_upper_right_uid[ipatch]};
      const std::vector<Int64>& cellnode_upper_left_uid{m_patch_cellnode_upper_left_uid[ipatch]};
      const std::vector<Int64>& cellnode_lower_right_uid{m_patch_cellnode_lower_right_uid[ipatch]};
      const std::vector<Int64>& cellnode_lower_left_uid{m_patch_cellnode_lower_left_uid[ipatch]};

      CellDirectionMng cell_dm{(patch_i == -1) ? m_cartesian_mesh->cellDirection(MD_DirX)
                                               : cm_patch->cellDirection(MD_DirX)};
      const CellGroup& allCellDMCells{cell_dm.allCells()};
      ENUMERATE_CELL (cell_i, allCellDMCells) {
        const Int64 i{cell_i.index()};
        const Cell& cell{*cell_i};

        {
          const Node& node_upper_right{cc.upperRight(cell)};
          ASSERT_EQUAL(cellnode_upper_right_uid[i], node_upper_right.uniqueId().asInt64());
        }
        {
          const Node& node_upper_left{cc.upperLeft(cell)};
          ASSERT_EQUAL(cellnode_upper_left_uid[i], node_upper_left.uniqueId().asInt64());
        }
        {
          const Node& node_lower_right{cc.lowerRight(cell)};
          ASSERT_EQUAL(cellnode_lower_right_uid[i], node_lower_right.uniqueId().asInt64());
        }
        {
          const Node& node_lower_left{cc.lowerLeft(cell)};
          ASSERT_EQUAL(cellnode_lower_left_uid[i], node_lower_left.uniqueId().asInt64());
        }
      }
    }

    {
      const Int32 ipatch{patch_i + 1};
      const Int32 lvl{(patch_i < 1) ? 0 : (patch_i < 3) ? 1 : 2};

      const std::vector<Int64>& nodecell_upper_right_uid{m_patch_nodecell_upper_right_uid[ipatch]};
      const std::vector<Int64>& nodecell_upper_left_uid{m_patch_nodecell_upper_left_uid[ipatch]};
      const std::vector<Int64>& nodecell_lower_right_uid{m_patch_nodecell_lower_right_uid[ipatch]};
      const std::vector<Int64>& nodecell_lower_left_uid{m_patch_nodecell_lower_left_uid[ipatch]};

      NodeDirectionMng node_dm{(patch_i == -1) ? m_cartesian_mesh->nodeDirection(MD_DirX)
                                               : cm_patch->nodeDirection(MD_DirX)};
      const NodeGroup& allNodeDMNodes{node_dm.allNodes()};
      ENUMERATE_NODE (node_i, allNodeDMNodes) {
        const Int64 i{node_i.index()};
        const Node node{*node_i};
        info() << " Node: " << ItemPrinter(node) << " patch=" << patch_i;
        {
          Cell cell_upper_right{(patch_i == -1) ? cc.upperRight(node) : upperRight(node, cm_patch, lvl)};
          info() << " CellUpperRight: " << ItemPrinter(cell_upper_right) << " expected=" << nodecell_upper_right_uid[i];
          ASSERT_EQUAL(nodecell_upper_right_uid[i], cell_upper_right.uniqueId().asInt64());
        }
        {
          const Cell& cell_upper_left{(patch_i == -1) ? cc.upperLeft(node) : upperLeft(node, cm_patch, lvl)};
          ASSERT_EQUAL(nodecell_upper_left_uid[i], cell_upper_left.uniqueId().asInt64());
        }
        {
          const Cell& cell_lower_right{(patch_i == -1) ? cc.lowerRight(node) : lowerRight(node, cm_patch, lvl)};
          ASSERT_EQUAL(nodecell_lower_right_uid[i], cell_lower_right.uniqueId().asInt64());
        }
        {
          const Cell& cell_lower_left{(patch_i == -1) ? cc.lowerLeft(node) : lowerLeft(node, cm_patch, lvl)};
          ASSERT_EQUAL(nodecell_lower_left_uid[i], cell_lower_left.uniqueId().asInt64());
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_UNITTESTCARTESIANMESHPATCH(UnitTestCartesianMeshPatch, UnitTestCartesianMeshPatchService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
} // namespace ArcaneTest
