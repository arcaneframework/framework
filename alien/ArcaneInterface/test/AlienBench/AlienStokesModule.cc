#include <mpi.h>

#include <string>
#include <map>
#include <time.h>
#include <vector>
#include <fstream>

#include <arcane/ArcaneVersion.h>
#include <arcane/Timer.h>
#include <arcane/ItemPairGroup.h>
#include <arcane/mesh/ItemFamily.h>
#include <arcane/utils/PlatformUtils.h>
#include <arcane/utils/IMemoryInfo.h>
#include <arcane/utils/OStringStream.h>
#include <arcane/ITimeLoopMng.h>
#include <alien/arcane_tools/accessors/ItemVectorAccessor.h>
#include <alien/core/block/VBlock.h>

#include <alien/arcane_tools/IIndexManager.h>
#include <alien/arcane_tools/indexManager/BasicIndexManager.h>
#include <alien/arcane_tools/indexManager/SimpleAbstractFamily.h>
#include <alien/arcane_tools/distribution/DistributionFabric.h>
#include <alien/arcane_tools/indexSet/IndexSetFabric.h>
#include <alien/arcane_tools/data/Space.h>

#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>

#include <alien/ref/AlienRefSemantic.h>

#include <alien/ref/mv_expr/MVExpr.h>

#ifdef ALIEN_USE_PETSC
#include <alien/kernels/petsc/io/AsciiDumper.h>
#include <alien/kernels/petsc/algebra/PETScLinearAlgebra.h>
#endif
#ifdef ALIEN_USE_MTL4
#include <alien/kernels/mtl/algebra/MTLLinearAlgebra.h>
#endif
#ifdef ALIEN_USE_HTSSOLVER
#include <alien/kernels/hts/HTSBackEnd.h>
#include <alien/kernels/hts/data_structure/HTSMatrix.h>
#include <alien/kernels/hts/algebra/HTSLinearAlgebra.h>
#endif
#ifdef ALIEN_USE_TRILINOS
#include <alien/kernels/trilinos/TrilinosBackEnd.h>
#include <alien/kernels/trilinos/data_structure/TrilinosMatrix.h>
#include <alien/kernels/trilinos/algebra/TrilinosLinearAlgebra.h>
#endif

#include <alien/expression/solver/ILinearSolver.h>
#include <alien/ref/AlienImportExport.h>

#include "AlienStokesModule.h"

#include <arcane/ItemPairGroup.h>
#include <arcane/IMesh.h>

#include <alien/core/impl/MultiVectorImpl.h>

using namespace Arcane;
using namespace Alien;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
Integer
AlienStokesModule::_computeDir(Real3 const& x)
{
  Integer dir = -1;
  Real max_coord = 0.;
  for (Integer i = 0; i < dim; ++i) {
    if (std::abs(x[i]) > max_coord) {
      dir = i;
      max_coord = std::abs(x[i]);
    }
  }
  assert(dir != -1);
  return dir;
}

Integer
AlienStokesModule::_computeOrientation(Face const& face)
{
  Integer dir = m_face_type[face];
  Integer orientation = 0;
  if (face.frontCell().null()) {
    Real3 n = m_face_center[face] - m_cell_center[face.backCell()];
    if (n[dir] > 0)
      orientation = 1;
    else
      orientation = -1;
  } else {
    Real3 n = m_cell_center[face.frontCell()] - m_face_center[face];
    if (n[dir] > 0)
      orientation = 1;
    else
      orientation = -1;
  }
  return orientation;
}

void
AlienStokesModule::init()
{
  Alien::setTraceMng(traceMng());
  Alien::setVerbosityLevel(Alien::Verbosity::Debug);
  m_parallel_mng = subDomain()->parallelMng();

  m_homogeneous = options()->homogeneous();

  for (Integer i = 0; i < dim; ++i)
    m_h[i] = 0.;

  ENUMERATE_CELL (icell, allCells()) {
    Real3 x;
    for (Arcane::Node node : icell->nodes()) {
      x += m_node_coord[node];
    }
    x /= icell->nbNode();
    m_cell_center[icell] = x;
  }

  ENUMERATE_FACE (iface, allFaces()) {
    Real3 x;
    for (Arcane::Node node : iface->nodes()) {
      x += m_node_coord[node];
    }
    x /= iface->nbNode();
    m_face_center[iface] = x;
    auto cell = iface->cell(0);
    Real3 n = m_cell_center[cell] - m_face_center[iface];
    Integer dir = _computeDir(n);
    m_face_type[iface] = dir;

    Integer orientation = _computeOrientation(*iface);
    m_face_orientation[iface] = orientation;

    if (m_h[dir] == 0 && iface->nbCell() == 2) {
      Real3 d = m_cell_center[iface->frontCell()] - m_cell_center[iface->backCell()];
      m_h[dir] = d.normL2();
      m_h2[dir] = m_h[dir] * m_h[dir];
    }
  }

  info() << " H  = " << m_h[0] << " " << m_h[1] << " " << m_h[2];
  info() << " H2 = " << m_h2[0] << " " << m_h2[1] << " " << m_h2[2];

  initSourceAndBoundaryTerm();

  Alien::ILinearSolver* solver = options()->linearSolver();
  solver->init();
}

/*---------------------------------------------------------------------------*/
bool
AlienStokesModule::_isNodeOfFace(Face const& face, Integer node_lid)
{
  for (Arcane::Node node : face.nodes()) {
    if (node.localId() == node_lid)
      return true;
  }
  return false;
}

void
AlienStokesModule::_computeFaceConnectivity(
    Face const& face, Alien::UniqueArray2<Integer>& connectivity)
{
  Integer face_lid = face.localId();
  Integer face_type = m_face_type[face];
  connectivity.fill(-1);
  Integer nb_nodes = face.nbNode();
  for (Integer i = 0; i < nb_nodes; ++i) {
    Node node0 = face.node(i);
    Node node1 = face.node(i == nb_nodes - 1 ? 0 : i + 1);
    Integer node1_lid = node1.localId();
    Real3 xN = 0.5 * (m_node_coord[node0] + m_node_coord[node1]) - m_face_center[face];
    Integer dir = _computeDir(xN);
    Integer up = xN[dir] > 0 ? 1 : 0;
    for (Arcane::Face conn_face : node0.faces()) {
      if (conn_face.localId() != face_lid && _isNodeOfFace(conn_face, node1_lid)
          && m_face_type[conn_face] == face_type) {
        connectivity[dir][up] = conn_face.localId();
      }
    }
    if (connectivity[dir][up] == -1) {
      connectivity[dir][up] = -1 - i;
    }
  }
}

Integer
AlienStokesModule::_upStreamFace(Face const& face, Integer cell_id)
{
  Integer up = 0;
  Integer orientation = m_face_orientation[face];
  if (orientation > 0) {
    if (!face.backCell().null()) {
      if (face.backCell().localId() == cell_id)
        up = 1;
      else
        up = -1;
    } else {
      if (face.frontCell().localId() == cell_id)
        up = -1;
      else
        up = 1;
    }
  } else {
    if (!face.backCell().null()) {
      if (face.backCell().localId() == cell_id)
        up = -1;
      else
        up = 1;
    } else {
      if (face.frontCell().localId() == cell_id)
        up = 1;
      else
        up = -1;
    }
  }
  assert(up != 0);
  return up;
}

AlienStokesModule::eBCType
AlienStokesModule::getBCType(Real3 const& xF, Integer dir)
{
  switch (dir) {
  case 0:
    if (xF[0] == 1.)
      return Neumann;
    else
      return Dirichlet;
  case 1:
    if (xF[0] == 1.)
      return Neumann;
    else
      return Dirichlet;
  case 2:
    if (xF[0] == 1.)
      return Neumann;
    else
      return Dirichlet;
  }
  return Dirichlet;
}
AlienStokesModule::eBCType
AlienStokesModule::getBCType(Face const& face)
{
  Integer face_type = m_face_type[face];
  Real3 xF = m_face_center[face];
  return getBCType(xF, face_type);
}
void
AlienStokesModule::initVelocityAndPressure()
{
  m_xV.resize(3);
  ENUMERATE_CELL (icell, allCells()) {
    Real3 const& xC = m_cell_center[icell];
    m_xV[icell][0] = ux(xC);
    m_xV[icell][1] = uy(xC);
    m_xV[icell][2] = uz(xC);
    m_xP[icell] = pressure(xC);
    m_xE[icell] = m_xV[icell][0] * m_xV[icell][0] + m_xV[icell][1] * m_xV[icell][1]
        + m_xV[icell][2] * m_xV[icell][2];
  }
}
void
AlienStokesModule::initSourceAndBoundaryTerm()
{
  ENUMERATE_CELL (icell, allCells()) {
    m_g[icell] = div(m_cell_center[*icell]);
  }

  ENUMERATE_FACE (iface, allFaces()) {
    m_f[iface] = func(m_face_center[*iface], m_face_type[*iface]);
  }

  ENUMERATE_FACE (iface, allCells().outerFaceGroup()) {
    m_fD[iface] = funcD(m_face_center[*iface], m_face_type[*iface]);
    m_fN[iface] = funcN(m_face_center[*iface], m_face_type[*iface]);
    m_flux[iface] = funcD(m_face_center[*iface], m_face_type[*iface]);
  }
}

void
AlienStokesModule::test()
{
  Timer pbuild_timer(subDomain(), "PBuildPhase", Timer::TimerReal);

  ItemGroup areaP = allCells();
  ItemGroup areaU = allCells().innerFaceGroup();

  ///////////////////////////////////////////////////////////////////////////
  //
  // CREATE GLOBAL Space FROM IndexManger
  // CREATE MATRIX ASSOCIATED TO Space
  // CREATE VECTORS ASSOCIATED TO Space
  //

  Alien::ArcaneTools::BasicIndexManager index_manager(m_parallel_mng);
  index_manager.setTraceMng(traceMng());
  auto indexSetU = index_manager.buildScalarIndexSet("U", areaU);
  auto indexSetP = index_manager.buildScalarIndexSet("P", areaP);
  index_manager.prepare();
  info() << "U Size : " << indexSetU.getAllIndexes().size();
  info() << "P Size : " << indexSetP.getAllIndexes().size();

  // Accès à l'indexation
  Alien::UniqueArray<Integer> allPIndex = index_manager.getIndexes(indexSetP);
  Alien::UniqueArray<Integer> allUIndex = index_manager.getIndexes(indexSetU);

  Alien::ArcaneTools::Space space(&index_manager);
  m_mdist = Alien::ArcaneTools::createMatrixDistribution(space);
  m_vdist = Alien::ArcaneTools::createVectorDistribution(space);

  info() << "Space size = " << m_vdist.globalSize()
         << ", local size= " << m_vdist.localSize();

  Alien::Matrix matrixA(m_mdist); // local matrix for exact measure without side effect
                                  // (however, you can reuse a matrix with several
                                  // builder)

  Alien::Vector vectorB(m_vdist);
  Alien::Vector vectorX(m_vdist);

  ///////////////////////////////////////////////////////////////////////////
  //
  // CREATE COMPOSITE Space FROM TWO IndexManger
  // CREATE MATRICES ASSOCIATED TO Composite Space USPace and PSpace
  // CREATE VECTORS ASSOCIATED TO USpace and PSpace
  //

  Alien::ArcaneTools::BasicIndexManager u_index_manager(m_parallel_mng);
  u_index_manager.setTraceMng(traceMng());
  auto u_indexSet = u_index_manager.buildScalarIndexSet("U", areaU);
  u_index_manager.prepare();
  info() << "U Size : " << u_indexSet.getAllIndexes().size();

  Alien::ArcaneTools::BasicIndexManager p_index_manager(m_parallel_mng);
  p_index_manager.setTraceMng(traceMng());
  auto p_indexSet = p_index_manager.buildScalarIndexSet("P", areaP);
  p_index_manager.prepare();
  info() << "P Size : " << p_indexSet.getAllIndexes().size();

  auto u_global_size = u_index_manager.globalSize();
  auto u_local_size = u_index_manager.localSize();
  auto p_global_size = p_index_manager.globalSize();
  auto p_local_size = p_index_manager.localSize();

  m_Adist = MatrixDistribution(
      u_global_size, u_global_size, u_local_size, parallelMng()->messagePassingMng());
  m_Bdist = MatrixDistribution(
      u_global_size, p_global_size, u_local_size, parallelMng()->messagePassingMng());
  m_tBdist = MatrixDistribution(
      p_global_size, u_global_size, p_local_size, parallelMng()->messagePassingMng());
  m_udist =
      VectorDistribution(u_global_size, u_local_size, parallelMng()->messagePassingMng());
  m_pdist =
      VectorDistribution(p_global_size, p_local_size, parallelMng()->messagePassingMng());

  Alien::Matrix A(m_Adist);
  Alien::Matrix B(m_Bdist);
  Alien::Matrix tB(m_tBdist);

  Alien::Vector g(m_pdist);
  Alien::Vector f(m_udist);

  Alien::UniqueArray<Integer> allPIndexB = p_index_manager.getIndexes(p_indexSet);
  Alien::UniqueArray<Integer> allUIndexA = u_index_manager.getIndexes(u_indexSet);

  ///////////////////////////////////////////////////////////////////////////
  //
  // MATRIX BUILDING AND FILLING
  //
  {
    Timer::Sentry ts(&pbuild_timer);
    Alien::MatrixProfiler profiler(matrixA);

    info() << "DEFINE BLOCK MATRIX PROFILE";
    Alien::MatrixProfiler profilerA(A);
    Alien::MatrixProfiler profilerB(B);
    Alien::MatrixProfiler profilertB(tB);

    ///////////////////////////////////////////////////////////////////////////
    //
    // DEFINE PROFILE
    //

    //
    // UX, UY, UZ equations
    //
    info() << "DEFINE BLOCK U-U PROFILE";

    UniqueArray2<Integer> connectivity(dim, 2);
    ENUMERATE_FACE (iface, areaP.innerFaceGroup().own()) {
      const Face& face = *iface;
      auto face_lid = iface->localId();
      auto face_type = m_face_type[face];
      info() << "FACE[" << face_lid << "]";
      const Integer iIndex = allUIndex[face.localId()];
      assert(iIndex != -1);
      profiler.addMatrixEntry(iIndex, allUIndex[face.localId()]);

      const Integer iIndexA = allUIndexA[face.localId()];
      assert(iIndexA != -1);

      profilerA.addMatrixEntry(iIndexA, allUIndexA[face.localId()]);

      for (Arcane::Cell cell : iface->cells()) {
        info() << "            CELL[" << cell.localId() << "]";
        assert(allPIndex[cell.localId()] != -1);
        profiler.addMatrixEntry(iIndex, allPIndex[cell.localId()]);

        profilerB.addMatrixEntry(iIndexA, allPIndexB[cell.localId()]);
        for (Arcane::Face face2 : cell.faces()){
          if (!face2.isSubDomainBoundary()) {
            if (face2.localId() != face_lid && m_face_type[face2] == face_type) {
              profiler.addMatrixEntry(iIndex, allUIndex[face2.localId()]);

              profilerA.addMatrixEntry(iIndexA, allUIndexA[face2.localId()]);
            }
          }
        }
      }

      _computeFaceConnectivity(face, connectivity);
      for (Integer i = 0; i < dim; ++i) {
        if (i != face_type) {
          for (Integer iconn = 0; iconn < 2; ++iconn) {
            if (connectivity[i][iconn] != -1) {
              assert(allUIndex[connectivity[i][iconn]] != -1);
              profiler.addMatrixEntry(iIndex, allUIndex[connectivity[i][iconn]]);

              profilerA.addMatrixEntry(iIndexA, allUIndexA[connectivity[i][iconn]]);
            }
          }
        }
      }
    }

    //
    // P equations
    //
    info() << "DEFINE BLOCK P-U PROFILE";
    ENUMERATE_CELL (icell, areaP.own()) {
      const Cell& cell = *icell;
      const Integer iIndex = allPIndex[cell.localId()];
      assert(iIndex != -1);
      info() << "CELL[" << cell.localId() << "]";
      if (m_cell_center[icell].normL2() < m_h[0]) {
        profiler.addMatrixEntry(iIndex, iIndex);
      }

      const Integer iIndexB = allPIndexB[cell.localId()];
      assert(iIndexB != -1);

      for (Arcane::Face face : icell->faces()) {
        info() << "        FACE[" << face.localId() << "]";
        if (!face.isSubDomainBoundary()) {
          assert(allUIndex[face.localId()] != -1);
          profiler.addMatrixEntry(iIndex, allUIndex[face.localId()]);

          assert(allUIndexA[face.localId()] != -1);
          profilertB.addMatrixEntry(iIndexB, allUIndexA[face.localId()]);
        }
      }
    }
  }

  {
    Timer::Sentry ts(&pbuild_timer);

    info() << "FILL MATRIX VALUE";
    Alien::ProfiledMatrixBuilder builder(
        matrixA, Alien::ProfiledMatrixOptions::eResetValues);

    info() << "FILL A FILLER";
    Alien::ProfiledMatrixBuilder builderA(A, Alien::ProfiledMatrixOptions::eResetValues);
    info() << "FILL B FILLER";
    Alien::ProfiledMatrixBuilder builderB(B, Alien::ProfiledMatrixOptions::eResetValues);
    info() << "FILL tB FILLER";
    Alien::ProfiledMatrixBuilder buildertB(
        tB, Alien::ProfiledMatrixOptions::eResetValues);

    info() << "INIT VECTORS";
    Alien::LocalVectorWriter rhs(vectorB);
    for (Integer i = 0; i < rhs.size(); ++i)
      rhs[i] = 0.;

    Alien::LocalVectorWriter rhsU(f);
    for (Integer i = 0; i < rhsU.size(); ++i)
      rhsU[i] = 0.;

    Alien::LocalVectorWriter rhsP(g);
    for (Integer i = 0; i < rhsP.size(); ++i)
      rhsP[i] = 0.;

    //
    // UX, UY, UZ equations
    //
    info() << "FILL U-U MATRIX VALUE";
    UniqueArray2<Integer> connectivity(dim, 2);
    ENUMERATE_FACE (iface, areaP.innerFaceGroup().own()) {
      const Face& face = *iface;
      if (face.isSubDomainBoundary())
        continue;

      auto face_lid = iface->localId();
      auto face_type = m_face_type[face];

      const Integer iIndex = allUIndex[face.localId()];
      assert(iIndex != -1);
      rhs[iIndex] += m_f[face];

      const Integer iIndexU = allUIndexA[face.localId()];
      assert(iIndexU != -1);
      rhsU[iIndexU] += m_f[face];

      Integer orientation = m_face_orientation[iface];

      Cell front_cell = iface->frontCell();
      if (!front_cell.null()) {
        builder(iIndex, allPIndex[front_cell.localId()]) = orientation / m_h[face_type];

        builderB(iIndexU, allPIndexB[front_cell.localId()]) =
            orientation / m_h[face_type];

        for (Arcane::Face face2 : front_cell.faces()) {
          if (face2.localId() != face_lid && m_face_type[face2] == face_type) {
            if (face2.isSubDomainBoundary()) {
              Integer bc_type = getBCType(face2);
              switch (bc_type) {
              case Neumann:
                rhs[iIndex] += m_fN[face2] / m_h[face_type];
                rhsU[iIndexU] += m_fN[face2] / m_h[face_type];

                break;
              case Dirichlet:
              default:
                rhs[iIndex] += m_fD[face2] / m_h2[face_type];
                builder(iIndex, iIndex) += 1. / m_h2[face_type];

                rhsU[iIndexU] += m_fD[face2] / m_h2[face_type];
                builderA(iIndexU, iIndexU) += 1. / m_h2[face_type];

                break;
              }
            } else {
              builder(iIndex, allUIndex[face2.localId()]) += -1 / m_h2[face_type];
              builder(iIndex, iIndex) += 1. / m_h2[face_type];

              builderA(iIndexU, allUIndexA[face2.localId()]) += -1 / m_h2[face_type];
              builderA(iIndexU, iIndexU) += 1. / m_h2[face_type];
            }
          }
        }
      }
      Cell back_cell = iface->backCell();
      if (!back_cell.null()) {
        builder(iIndex, allPIndex[back_cell.localId()]) =
            -1. * orientation / m_h[face_type];
        builderB(iIndexU, allPIndexB[back_cell.localId()]) =
            -1. * orientation / m_h[face_type];

        for (Arcane::Face face2 : back_cell.faces()) {
          if (face2.localId() != face_lid && m_face_type[face2] == face_type) {
            if (face2.isSubDomainBoundary()) {
              Integer bc_type = getBCType(face2);
              switch (bc_type) {
              case Neumann:
                rhs[iIndex] += m_fN[face2] / m_h[face_type];
                rhsU[iIndexU] += m_fN[face2] / m_h[face_type];

                break;
              case Dirichlet:
              default:
                rhs[iIndex] += m_fD[face2] / m_h2[face_type];
                builder(iIndex, iIndex) += 1. / m_h2[face_type];

                rhsU[iIndex] += m_fD[face2] / m_h2[face_type];
                builderA(iIndexU, iIndexU) += 1. / m_h2[face_type];

                break;
              }
            } else {
              builder(iIndex, allUIndex[face2.localId()]) += -1 / m_h2[face_type];
              builder(iIndex, iIndex) += 1. / m_h2[face_type];

              builderA(iIndexU, allUIndexA[face2.localId()]) += -1 / m_h2[face_type];
              builder(iIndexU, iIndexU) += 1. / m_h2[face_type];
            }
          }
        }
      }
      _computeFaceConnectivity(face, connectivity);
      for (Integer dir = 0; dir < dim; ++dir) {
        if (dir != face_type) {
          for (Integer iconn = 0; iconn < 2; ++iconn) {
            if (connectivity[dir][iconn] >= 0) {
              assert(allUIndex[connectivity[dir][iconn]] != -1);
              builder(iIndex, allUIndex[connectivity[dir][iconn]]) += -1 / m_h2[dir];
              builder(iIndex, iIndex) += 1. / m_h2[dir];

              builderA(iIndexU, allUIndexA[connectivity[dir][iconn]]) += -1 / m_h2[dir];
              builderA(iIndexU, iIndexU) += 1. / m_h2[dir];
            } else {
              Integer inode0 = -connectivity[dir][iconn] - 1;
              Node node0 = face.node(inode0);
              Real3 xN0 = m_node_coord[node0];
              Integer inode1 = inode0 == face.nbNode() - 1 ? 0 : inode0 + 1;
              Node node1 = face.node(inode1);
              Real3 xN1 = m_node_coord[node1];
              Integer bc_type = getBCType(xN0, dir);
              switch (bc_type) {
              case Neumann:
                rhs[iIndex] += 0.5 * (funcN(xN0, dir) + funcN(xN1, dir)) / m_h[dir];
                rhsU[iIndexU] += 0.5 * (funcN(xN0, dir) + funcN(xN1, dir)) / m_h[dir];
                break;
              case Dirichlet:
              default:
                rhs[iIndex] += (funcD(xN0, dir) + funcD(xN1, dir)) / m_h2[dir];
                builder(iIndex, iIndex) += 1. / m_h2[dir];

                rhsU[iIndexU] += (funcD(xN0, dir) + funcD(xN1, dir)) / m_h2[dir];
                builderA(iIndexU, iIndexU) += 1. / m_h2[dir];
                break;
              }
            }
          }
        }
      }
    }

    //
    // P equations
    //
    ENUMERATE_CELL (icell, areaP.own()) {
      const Cell& cell = *icell;
      Integer cell_lid = icell->localId();
      const Integer iIndex = allPIndex[cell.localId()];
      assert(iIndex != -1);

      const Integer iIndexP = allPIndexB[cell.localId()];
      assert(iIndexP != -1);

      if (m_cell_center[icell].normL2() < m_h[0])
        builder(iIndex, iIndex) = 1000000;

      rhs[iIndex] += m_g[cell];

      rhsP[iIndexP] += m_g[cell];

      for (Arcane::Face face : icell->faces()) {
        Integer face_lid = face.localId();
        Integer face_type = m_face_type[face];
        Integer up = _upStreamFace(face, cell_lid);
        if (face.isSubDomainBoundary()) {
          rhs[iIndex] -= up / m_h[face_type];
          rhsP[iIndexP] -= up / m_h[face_type];

          Integer bc_type = getBCType(face);
          info() << "     Boundary F" << bc_type;
          switch (bc_type) {
          case Neumann:
            rhs[iIndex] -= up * m_fN[face];
            rhsP[iIndexP] -= up * m_fN[face];
            for (Arcane::Face face2 : icell->faces()) {
              Integer face2_type = m_face_type[face2];
              Integer face2_lid = face2.localId();
              if (face2_lid != face_lid && face2_type == face_type) {
                builder(iIndex, allUIndex[face2.localId()]) += up / m_h[face_type];
                buildertB(iIndexP, allUIndexA[face2.localId()]) += up / m_h[face_type];
              }
            }
            break;
          case Dirichlet:
          default:
            rhs[iIndex] -= up * m_fD[face] / m_h[face_type];
            rhsP[iIndexP] -= up * m_fD[face] / m_h[face_type];
            break;
          }
        } else {
          builder(iIndex, allUIndex[face.localId()]) += up / m_h[face_type];
          buildertB(iIndexP, allUIndexA[face.localId()]) += up / m_h[face_type];
        }
      }
    }

    info() << "BUILDER FINALIZE";

    builder.finalize();

    builderA.finalize();
    builderB.finalize();
    buildertB.finalize();
  }

  {
    SystemWriter writer("StokesMatrix", "ascii", parallelMng()->messagePassingMng());
    writer.dump(matrixA);
  }
  {
    SystemWriter writer("StokesMatrixA", "ascii", parallelMng()->messagePassingMng());
    writer.dump(A);
  }
  {
    SystemWriter writer("StokesMatrixB", "ascii", parallelMng()->messagePassingMng());
    writer.dump(B);
  }
  {
    SystemWriter writer("StokesMatrixtB", "ascii", parallelMng()->messagePassingMng());
    writer.dump(tB);
  }
  {
    std::ofstream fout("StokesVectorF.txt");
    Alien::VectorReader f_view(f);
    fout << f_view.size() << std::endl;
    for (int i = 0; i < f_view.size(); ++i) {
      fout << i << " " << f_view[i] << std::endl;
    }
  }
  {
    std::ofstream fout("StokesVectorG.txt");
    Alien::VectorReader g_view(g);
    fout << g_view.size() << std::endl;
    for (int i = 0; i < g_view.size(); ++i) {
      fout << i << " " << g_view[i] << std::endl;
    }
  }

  info() << "===================================================";
  info() << "STOKES INFO :";
  info() << " PBUILD    :" << pbuild_timer.totalTime();
  info() << "===================================================";

  ///////////////////////////////////////////////////////////////////////////
  //
  // RESOLUTION
  //
  bool solve = true;
  if (solve) {
    Alien::Vector p(m_pdist);
    Alien::Vector u(m_udist);

    // Initial guest
    Alien::LocalVectorWriter u0_view(u);
    for (Integer i = 0; i < u0_view.size(); ++i)
      u0_view[i] = 0.;

    Alien::LocalVectorWriter p0_view(p);
    for (Integer i = 0; i < p0_view.size(); ++i)
      p0_view[i] = 0.;

    bool succeed = solveUzawaMethod(A, B, tB, f, g, u, p);

    if (succeed) {
      Alien::VectorReader u_view(u);
      Alien::VectorReader p_view(p);

      ENUMERATE_FACE (iface, areaU.own()) {
        const Integer iIndex = allUIndexA[iface->localId()];
        m_flux[iface] = u_view[iIndex];
      }

      m_v.resize(dim);
      ENUMERATE_CELL (icell, areaP.own()) {
        const Integer iIndex = allPIndexB[icell->localId()];
        m_p[icell] = p_view[iIndex];
        Real v[dim] = { 0., 0., 0. };
        for (Arcane::Face face : icell->faces()){
          v[m_face_type[face]] += m_flux[face];
        }
        m_v[icell][0] = v[0] / 2;
        m_v[icell][1] = v[1] / 2;
        m_v[icell][2] = v[2] / 2;
        m_e[icell] = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) / 4;
      }
    }
  }

  subDomain()->timeLoopMng()->stopComputeLoop(true);
}

/*---------------------------------------------------------------------------*/

Real
AlienStokesModule::pressure(Real3 const& x) const
{
  return 1 - x[0];
}

Real
AlienStokesModule::ux(Real3 const& x) const
{
  return 1 + x[0];
}

Real
AlienStokesModule::duxdn([[maybe_unused]] Real3 const& x, Integer dir) const
{
  switch (dir) {
  case 0:
    return 1.;
  default:
    return 0.;
  }
}

Real
AlienStokesModule::uy(Real3 const& x) const
{
  return x[1] - 0.5;
}

Real
AlienStokesModule::duydn([[maybe_unused]] Real3 const& x, Integer dir) const
{
  switch (dir) {
  case 1:
    return 1.;
  default:
    return 0.;
  }
}

Real
AlienStokesModule::uz(Real3 const& x) const
{
  return x[2] - 0.5;
}

Real
AlienStokesModule::duzdn([[maybe_unused]] Real3 const& x, Integer dir) const
{
  switch (dir) {
  case 2:
    return 1.;
  default:
    return 0.;
  }
}

Real
AlienStokesModule::div([[maybe_unused]] Real3 const& xC) const
{
  return 3;
}

Real
AlienStokesModule::func([[maybe_unused]] Real3 const& xC, Integer dir) const
{
  switch (dir) {
  case 0:
    return 1;
  default:
    return 0.;
  }
}

Real
AlienStokesModule::funcN(Real3 const& xF, Integer dir) const
{
  switch (dir) {
  case 0:
    return duxdn(xF, dir);
  case 1:
    return duydn(xF, dir);
  case 2:
    return duzdn(xF, dir);
  default:
    return 0.;
  }
}

Real
AlienStokesModule::funcD(Real3 const& xF, Integer dir) const
{
  switch (dir) {
  case 0:
    return ux(xF);
  case 1:
    return uy(xF);
  case 2:
    return uz(xF);
  default:
    return 0.;
  }
}

bool
AlienStokesModule::solveUzawaMethod(Alien::Matrix& A, Alien::Matrix& B, Alien::Matrix& tB,
    Alien::Vector& f, Alien::Vector& g, Alien::Vector& uk, Alien::Vector& pk)
{
  Timer psolve_timer(subDomain(), "PSolvePhase", Timer::TimerReal);

  m_omega = options()->uzawaFactor();
  m_uzawa_max_nb_iterations = options()->uzawaMaxNbIterations();

  Alien::ILinearSolver* solver = options()->linearSolver();
  solver->init();

  Vector ru(m_udist);
  Vector rp(m_pdist);

  using namespace Alien;
  using namespace Alien::MVExpr;

  for (int k = 0; k < m_uzawa_max_nb_iterations; ++k) {
    // Update velocity
    ru = f - B * pk;

    {
      Timer::Sentry ts(&psolve_timer);
      solver->solve(A, ru, uk);
    }
    Alien::SolverStatus status = solver->getStatus();
    if (!status.succeeded)
      return false;

    rp = g - tB * uk;
    // Update pressure
    pk = pk - m_omega * rp;
  }

  solver->end();

  info() << "===================================================";
  info() << "STOKES INFO :";
  info() << " PSOLVE    :" << psolve_timer.totalTime();
  info() << "===================================================";

  return true;
}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_ALIENSTOKES(AlienStokesModule);
