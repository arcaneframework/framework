#include <mpi.h>
#include "MemoryAllocationTracker.h"

#include <boost/lexical_cast.hpp>
#include <boost/timer.hpp>
#include <time.h>
#include <vector>

#include <arcane/ArcaneVersion.h>
#include <arcane/ITimeLoopMng.h>
#include <arcane/Timer.h>
#include <arcane/mesh/ItemFamily.h>
#include <arcane/utils/IMemoryInfo.h>
#include <arcane/utils/OStringStream.h>
#include <arcane/utils/PlatformUtils.h>

#include <alien/AlienExternalPackages.h>

#include <alien/arcane_tools/accessors/ItemVectorAccessor.h>
#include <alien/arcane_tools/indexManager/BasicIndexManager.h>
#include <alien/arcane_tools/indexManager/SimpleAbstractFamily.h>
#include <alien/core/block/VBlock.h>

#include <alien/arcane_tools/IIndexManager.h>

#include <alien/move/handlers/block/BlockVectorReader.h>
#include <alien/move/handlers/block/BlockVectorWriter.h>
#include <alien/move/handlers/scalar/VectorReader.h>
#include <alien/move/handlers/scalar/VectorWriter.h>
#include <alien/arcane_tools/block/BlockSizes.h>
#include <alien/arcane_tools/block/BlockBuilder.h>
#include <alien/arcane_tools/distribution/DistributionFabric.h>
#include <alien/arcane_tools/indexSet/IndexSetFabric.h>
#include <alien/move/handlers/block/ProfiledBlockMatrixBuilder.h>
#include <alien/move/handlers/scalar/DirectMatrixBuilder.h>
#include <alien/move/handlers/scalar/MatrixProfiler.h>
#include <alien/move/handlers/scalar/ProfiledMatrixBuilder.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>
#ifdef ALIEN_USE_PETSC
#include <alien/kernels/petsc/io/AsciiDumper.h>
#include <alien/kernels/petsc/algebra/PETScLinearAlgebra.h>
#endif
#ifdef ALIEN_USE_MTL4
#include <alien/kernels/mtl/algebra/MTLLinearAlgebra.h>
#endif

#include <alien/expression/solver/ILinearSolver.h>

#include "AlienTestOptionTypes.h"
#include "AlienTestModule.h"

#include <arcane/ItemPairGroup.h>
#include <arcane/IMesh.h>

#include <alien/core/impl/MultiVectorImpl.h>

#include <alien/expression/krylov/AlienKrylov.h>

#include <alien/utils/StdTimer.h>


using namespace Arcane;
using namespace Alien;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
AlienTestModule::init()
{
  Alien::setTraceMng(traceMng());
  Alien::setVerbosityLevel(Alien::Verbosity::Debug);

  m_diag_coeff = options()->diagonalCoefficient();
  switch (options()->stencilBy()) {
  case AlienTestOptionTypes::StencilByNode:
    m_stencil_kind = IK_Node;
    break;
  case AlienTestOptionTypes::StencilByFace:
    m_stencil_kind = IK_Face;
    break;
  }

  m_vect_size = options()->vectSize();
  if (m_vect_size == 0) {
    m_areaT = allCells();

    m_split.fill(0);

    ENUMERATE_CELL (icell, m_areaT.own()) {
      if (m_uniform() < 0.9) {
        m_split[icell] = 1;
      }
    }

    m_split.synchronize();

    {
      Arcane::IntegerUniqueArray areaP_lids;
      areaP_lids.reserve(m_areaT.size());

      ENUMERATE_CELL (icell, m_areaT) {
        if (m_split[icell] == 1)
          areaP_lids.add(icell->localId());
      }
      m_areaP = mesh()->cellFamily()->createGroup("AreaP", areaP_lids);
    }

    m_P.fill(0);
    ENUMERATE_CELL (icell, m_areaP)
      m_P[icell] = 1.;

    m_T.fill(0);
    ENUMERATE_CELL (icell, m_areaT)
      m_T[icell] = 1.;

    m_split.fill(0);
    ENUMERATE_CELL (icell, m_areaP)
      m_split[icell] = 1;
  }

  m_parallel_mng = subDomain()->parallelMng();

  if (m_parallel_mng->commRank() % 2 == 0)
    m_n_extra_indices = options()->extraEquationCount();
  else
    m_n_extra_indices = 0;

  for (int i = 0; i < options()->linearSolver.size(); ++i) {
    Alien::ILinearSolver* solver = options()->linearSolver[i];
    solver->init();
  }
}

/*---------------------------------------------------------------------------*/

void
AlienTestModule::test()
{
  Timer timer(subDomain(), "AlienTestModule", Timer::TimerVirtual);
  if (m_vect_size == 1)
    m_w.resize(2);
  else if (m_vect_size > 1)
    m_w.resize(m_vect_size);

  if (options()->checkMemory()) {
    m_memory_tracker = new MemoryAllocationTracker();
    m_memory_tracker->beginCollect();
  }

  ItemGroup areaU = allCells();
  // ItemGroup areaU = mesh()->cellFamily()->createGroup("AreaU");
  ItemGroup areaP = allNodes();
  // ItemGroup areaP = mesh()->nodeFamily()->createGroup("AreaP");
  CellCellGroup cell_cell_connection(areaU.own(), areaU, m_stencil_kind);

  BuildingStat index_manager_stat(this, timer);
  index_manager_stat.start();

  Alien::ArcaneTools::BasicIndexManager index_manager(m_parallel_mng);
  index_manager.setTraceMng(traceMng());

  // index_manager.init() ; // facultatif après une construction; principalement utilisé
  // pour une réutilisation de l'objet

  // Area doit répondre est une collection énumérable où chaque élément doit répondre à
  // 'localId', 'uniqueId' et 'owner'
  // 'localId' est pour l'accès au table 'cache' d'indexation et 'uniqueId' et 'owner'
  // pour les algorthmiques d'indexation à proprement dit.
  // L'indexation s'occupe d'indexer des entités (le plus souvent des items)
  // indépendamment des concepts de système linéaire.
  // Ainsi l'indexation est un objet qui forme une bijection entre des entités et un
  // intervalle [O:N[ (par bloc en parallèle)
  auto indexSetU = index_manager.buildScalarIndexSet("U", areaU);
  // la variable U est taguee en "Elliptic" pour un traitement de type CprAMG:
  // NOTE: le tag "block-tag" et la valeur "Elliptic" sont pre-definis specifiquement pour
  // le cas CprAMG de IFPSolver
  indexSetU.addTag("block-tag", "Elliptic");

  // Idem en vectoriel
  auto indexSetP = index_manager.buildVectorIndexSet("P", areaP, 2);
  // Si l'on veut mettre des tags, on peut les mettre par composante ou pour tout l'entité
  // vectorielle
  indexSetP[0].addTag("block-tag", "P");
  indexSetP[1].addTag("block-tag", "Q");

  Arcane::Int64UniqueArray
      xUids; // ajout de quelques indices extras pour les processeurs de rank pair
  const Integer max_n_extra_indices =
      m_parallel_mng->reduce(Arcane::Parallel::ReduceMax, m_n_extra_indices);
  for (Integer i = 0; i < m_n_extra_indices; ++i) {
    xUids.add(m_parallel_mng->commRank() * max_n_extra_indices + i);
    info() << "xUids(" << i << ") : " << xUids[i];
  }
  auto indexSetX = index_manager.buildScalarIndexSet(
      "X", Alien::ArcaneTools::SimpleAbstractFamily(xUids, &index_manager));
  indexSetX.addTag("block-tag", "P");
  index_manager.prepare();

  ///////////////////////////////////////////////////////////////////////////
  //
  // CREATE Space FROM IndexManger
  // CREATE MATRIX ASSOCIATED TO Space
  // CREATE VECTORS ASSOCIATED TO Space
  //

  // Accès à l'indexation
  Arccore::UniqueArray<Integer> allUIndex = index_manager.getIndexes(indexSetU);
  UniqueArray2<Integer> allPIndex = index_manager.getIndexes(indexSetP);
  Arccore::UniqueArray<Integer> allXIndex = index_manager.getIndexes(indexSetX);

  m_mdist = Alien::ArcaneTools::createMatrixDistribution(&index_manager, parallelMng()->messagePassingMng());
  m_vdist = Alien::ArcaneTools::createVectorDistribution(&index_manager, parallelMng()->messagePassingMng());

  Alien::Space space(m_vdist.globalSize(), "TestSpace");

  // Ajout des champs à l'espace (pour FieldSplit)
  Alien::ArcaneTools::createIndexSet(space, &index_manager, "Elliptic", traceMng());
  Alien::ArcaneTools::createIndexSet(space, &index_manager, "P", traceMng());
  Alien::ArcaneTools::createIndexSet(space, &index_manager, "Q", traceMng());

  Alien::Move::VectorData vectorB(space, m_vdist);
  Alien::Move::VectorData vectorX(space, m_vdist);

  std::unique_ptr<Alien::ArcaneTools::BlockBuilder> block_builder(nullptr);
  std::unique_ptr<Alien::VBlock> vblock(nullptr);
  if (m_vect_size == 0) {
    block_builder.reset(
        new Alien::ArcaneTools::BlockBuilder(index_manager));
    auto& builder = *(block_builder.get());
    {
      auto areaT_lids = m_areaT.own().view().localIds();
      UniqueArray<Integer> sample_areaT_lids(areaT_lids.size());
      allUIndex.sample(areaT_lids, sample_areaT_lids);
      builder[sample_areaT_lids] = 1;
    }
    {
      auto areaP_lids = m_areaP.own().view().localIds();
      UniqueArray<Integer> sample_areaP_lids(areaP_lids.size());
      allUIndex.sample(areaP_lids, sample_areaP_lids);
      builder[sample_areaP_lids] += 3;
    }
    ConstArray2View<Integer> p_indexes = allPIndex;
    ENUMERATE_NODE (inode, areaP.own()) {
      Integer index0 = p_indexes[inode->localId()][0];
      Integer index1 = p_indexes[inode->localId()][1];
      builder[index0] = 1;
      builder[index1] = 1;
    }
    for (Integer i = 0; i < m_n_extra_indices; ++i) {
      builder[allXIndex[i]] = 1;
      info() << "index of xUids(" << i << ") : " << allXIndex[i];
    }
    //builder.compute();
    vblock.reset(new Alien::VBlock(builder.sizes()));
    vectorB.setBlockInfos(vblock.get());
    vectorX.setBlockInfos(vblock.get());
  } else if (m_vect_size > 1) {
    vectorB.setBlockInfos(m_vect_size);
    vectorX.setBlockInfos(m_vect_size);
  }

  index_manager_stat.stop();

  ///////////////////////////////////////////////////////////////////////////
  //
  // VECTOR BUILDING AND FILLING
  //
  info() << "Building & initializing vector b";
  info() << "Space size = " << m_vdist.globalSize()
         << ", local size= " << m_vdist.localSize();
  if (m_vect_size == 0) {
    UniqueArray<Real> values(4);
    values.fill(2.);
    buildAndFillInBlockVector(vectorB, allUIndex, allPIndex, allXIndex, values);
  } else if (m_vect_size == 1)
    buildAndFillInVector(vectorB, 2.);
  else
    buildAndFillInBlockVector(vectorB, 2.);

// Testing linear algebra on vectors
#ifdef ALIEN_USE_PETSC
  {
    Alien::PETScLinearAlgebra algebra2;
    algebra2.copy(vectorB, vectorX); // copy b in x using petsc
    info() << "DOT " << algebra2.dot(vectorB, vectorX);
  }
#endif // ALIEN_USE_PETSC

  info() << "Checking vector x data";
  if (m_vect_size > 0) {
    checkVectorValues(vectorX, 2.);
  }

// Dump an object
#ifdef ALIEN_USE_PETSC
  {
    std::cout << "AVANT" << std::endl;
    // Alien::AsciiDumper dumper;
    Alien::AsciiDumper dumper(Alien::AsciiDumper::eMatlabStyle);
    std::cout << "vectorX" << std::endl;
    dumper.dump("vectorX.txt", vectorX);
    std::cout << "vectorB" << std::endl;
    dumper.dump("vectorB.txt", vectorB);
    std::cout << "APRES" << std::endl;
  }
#endif // ALIEN_USE_PETSC

  ///////////////////////////////////////////////////////////////////////////
  //
  // SCALAR PRODUCT
  //
  info() << "Checking dot product using different linear algebra implementations";
  checkDotProductWithManyAlgebra(vectorB, vectorX, space);

  ///////////////////////////////////////////////////////////////////////////
  //
  // VECTOR / VARIABLE ASSOCIATION
  //
  if (m_vect_size > 0) {
    vectorVariableUpdate(vectorB, indexSetU);
    info() << "Checking vector b data after a reverse commit";
    checkVectorValues(vectorB, 2.);
  }

  Arcane::XmlNode rootXmlBuilderNode = options()->builder.rootElement();
  Arcane::XmlNodeList builderNodeList = rootXmlBuilderNode.children("builder");

  Alien::Move::MatrixData matrixA(space, m_mdist); // local matrix for exact measure without
                                             // side effect (however, you can reuse a
                                             // matrix with several builder)
  if (m_vect_size == 0) {
    matrixA.setBlockInfos(vblock.get());
  } else if (m_vect_size > 1) {
    matrixA.setBlockInfos(m_vect_size);
  }
  for (Integer ibuilder = 0; ibuilder < options()->builder.size(); ++ibuilder) {

    info() << "BUILDER #" << ibuilder << " : " << builderNodeList[ibuilder].value();

    BuildingStat profile_stat(this, timer);
    UniqueArray<BuildingStat> inserter_stats;

    ///////////////////////////////////////////////////////////////////////////
    //
    // MATRIX BUILDING AND FILLING
    //
    {
      AlienTestOptionTypes::eBuilderType builder_type = options()->builder[ibuilder];
      if (builder_type == AlienTestOptionTypes::DirectBuilder && m_vect_size != 1)
        continue;
      if (m_vect_size == 0 && builder_type != AlienTestOptionTypes::StreamBuilder)
        continue;
      if (builder_type == AlienTestOptionTypes::StreamBuilder)
        continue; // XT (28/06/2016) : no more test in that framework for StreamBuilder
                  // (as it won't work the same way than other builders)
      switch (builder_type) {
      case AlienTestOptionTypes::DirectBuilder: {
        if (m_vect_size == 1)
          for (Integer loop = 0; loop < options()->repeatLoop(); ++loop) {
            info() << "Filling #" << loop;
            Alien::Move::DirectMatrixBuilder builder(
                std::move(matrixA), Alien::DirectMatrixOptions::eResetValues);
            if (loop == 0) {
              profile_stat.start();
              builder.reserve(indexSetU.getOwnIndexes(), 30);
              builder.reserve(indexSetP[0].getOwnIndexes(), 1);
              builder.reserve(indexSetP[1].getOwnIndexes(), 1);
              builder.reserve(indexSetX.getOwnIndexes(), 1);
              builder.allocate(); // on force le placement de allocate ici pour les
                                  // mesures de perfs (normalement seul l'autre suffit)
              profile_stat.stop();
            }
            inserter_stats.add(BuildingStat(this, timer));
            if (loop != 0)
              builder.allocate();
            inserter_stats.back().start();
            fillInMatrix(
                cell_cell_connection, areaP, allPIndex, allUIndex, allXIndex, builder);
            // info() << builder.stats(&index_manager);
            builder
                .finalize(); // process extra work after filling (squeeze, non-local...)
            // optional if used with out-of-scope dtor
            inserter_stats.back().stop();
            matrixA = builder.release();
          }
      } break;
      case AlienTestOptionTypes::ProfiledBuilder: {
        profile_stat.start();
        {
          Alien::Move::MatrixProfiler profiler(std::move(matrixA));
          profileMatrix(
              cell_cell_connection, areaP, allPIndex, allUIndex, allXIndex, profiler);
          profiler.allocate(); // optional if used with out-of-scope dtor
          matrixA = profiler.release();
        }
        profile_stat.stop();

        for (Integer loop = 0; loop < options()->repeatLoop(); ++loop) {
          info() << "Filling #" << loop;
          inserter_stats.add(BuildingStat(this, timer));
          inserter_stats.back().start();
          if (m_vect_size == 0) {

          } else if (m_vect_size == 1) {
            Alien::Move::ProfiledMatrixBuilder builder(
                std::move(matrixA), Alien::ProfiledMatrixOptions::eResetValues);
            fillInMatrix(
                cell_cell_connection, areaP, allPIndex, allUIndex, allXIndex, builder);
            builder
                .finalize(); // process extra work after filling (squeeze, non-local...)
            // optional if used with out-of-scope dtor
            matrixA = builder.release();

            {
              SimpleCSRLinearAlgebra alg;
              Alien::Move::VectorData vectorR(space, m_vdist);
              vectorR.setBlockInfos(vectorX.block());
              alg.mult(matrixA, vectorX, vectorR);
              alg.axpy(-1., vectorB, vectorR);
              Real res = alg.norm2(vectorR);
              info() << "CSR RES : " << res;
            }
#ifdef ALIEN_USE_MTL4
            if (m_parallel_mng->commSize() == 1) {
              Alien::MTLLinearAlgebra alg;
              Alien::Move::VectorData vectorR(space, m_vdist);
              vectorR.setBlockInfos(vectorX.block());
              alg.mult(matrixA, vectorX, vectorR);
              alg.axpy(-1., vectorB, vectorR);
              Real res = alg.norm2(vectorR);
              info() << "MTL RES : " << res;
            }
#endif
          } else if (m_vect_size > 1) {
            Alien::Move::ProfiledBlockMatrixBuilder builder(std::move(matrixA),
                Alien::ProfiledBlockMatrixBuilderOptions::eResetValues);
            profiledFillInMatrix(
                cell_cell_connection, areaP, allPIndex, allUIndex, allXIndex, builder);
            builder
                .finalize(); // process extra work after filling (squeeze, non-local...)
            matrixA = builder.release();
          }
          inserter_stats.back().stop();
        }
      } break;

      /*
      case AlienTestOptionTypes::StreamBuilder:
      {
        profile_stat.start();
        if(m_vect_size==0)
        {
          //jmg Alien::StreamBlockMatrixBuilder stream_builder(std::move(matrixA),
      block_builder.get());
          Alien::StreamBlockMatrixBuilder stream_builder(matrixA, block_builder.get());
          // stream_builder.init() ; // optional : already done by ctor (only used by
      multiple usage of MatrixBuilder)
          // stream_builder.start() ;
          streamProfileMatrix(cell_cell_connection, areaU, areaP, allPIndex, allUIndex,
      allXIndex, stream_builder);

          profile_stat.stop();

          for(Integer loop = 0; loop<options()->repeatLoop();++loop)
          {
            info() << "Filling #" << loop;
            stream_builder.fillZero();

            inserter_stats.add(BuildingStat(this,timer));
            inserter_stats.back().start();

            streamBlockFillInMatrix(cell_cell_connection, areaP, allPIndex, allUIndex,
      allXIndex, stream_builder);
            stream_builder.finalize(); // process extra work after filling (squeeze,
      non-local...)
            // NOT optional ! (since this object is usually used in a loop)
            inserter_stats.back().stop();
          }
          //jmg matrixA = stream_builder.release();
        }
        else if(m_vect_size==1)
        {
          Alien::StreamMatrixBuilder stream_builder(matrixA);
          // stream_builder.init() ; // optional : already done by ctor (only used by
      multiple usage of MatrixBuilder)
          // stream_builder.start() ;
          streamProfileMatrix(cell_cell_connection, areaU, areaP, allPIndex, allUIndex,
      allXIndex, stream_builder);

          profile_stat.stop();

          for(Integer loop = 0; loop<options()->repeatLoop();++loop)
          {
            info() << "Filling #" << loop;
            stream_builder.fillZero();

            inserter_stats.add(BuildingStat(this,timer));
            inserter_stats.back().start();

            streamFillInMatrix(cell_cell_connection, areaP, allPIndex, allUIndex,
      allXIndex, stream_builder);
            stream_builder.finalize(); // process extra work after filling (squeeze,
      non-local...)
            // NOT optional ! (since this object is usually used in a loop)
            inserter_stats.back().stop();
          }
          //matrixA = stream_builder.release();
        }
        else if(m_vect_size>1)
        {
          Alien::StreamMatrixBuilder stream_builder(matrixA);
          // stream_builder.init() ; // optional : already done by ctor (only used by
      multiple usage of MatrixBuilder)
          // stream_builder.start() ;
          streamProfileMatrix(cell_cell_connection, areaU, areaP, allPIndex, allUIndex,
      allXIndex, stream_builder);

          profile_stat.stop();

          for(Integer loop = 0; loop<options()->repeatLoop();++loop)
          {
            info() << "Filling #" << loop;
            stream_builder.fillZero();

            inserter_stats.add(BuildingStat(this,timer));
            inserter_stats.back().start();

            streamBlockFillInMatrix(cell_cell_connection, areaP, allPIndex, allUIndex,
      allXIndex, stream_builder);
            stream_builder.finalize(); // process extra work after filling (squeeze,
      non-local...)
            // NOT optional ! (since this object is usually used in a loop)
            inserter_stats.back().stop();
          }
          //matrixA = stream_builder.release();
        }

        // stream_builder.end(); // optional : already done by dtor
      }
      break;
      */
      default:
        fatal() << "Builder type : " << builder_type << " Not yet available";
        break;
      }
    }
    ///////////////////////////////////////////////////////////////////////////
    //
    // RESOLUTION
    //
    Arcane::XmlNode rootXmlNode = options()->linearSolver.rootElement();
    Arcane::XmlNodeList linearSolverNodeList = rootXmlNode.children("linear-solver");

    UniqueArray<UniqueArray<Real>> solverData;

    if (m_vect_size == 0) {
      Alien::SimpleCSRLinearAlgebra testAlg;
      Alien::Move::VectorData vectorY(space, m_vdist);
      Alien::Move::VectorData vectorRes(space, m_vdist);

      vectorY.setBlockInfos(vectorX.block());
      vectorRes.setBlockInfos(vectorX.block());

      testAlg.mult(matrixA, vectorX, vectorY);
      // res = b - A.x
      testAlg.copy(vectorB, vectorRes);
      testAlg.axpy(-1., vectorY, vectorRes);
      Real norm = testAlg.norm2(vectorRes);
      info() << "Residual norm2 by VBlockCSR algebra : " << norm;
    } else if (!options()->buildingOnly())
      for (int i = 0; i < options()->linearSolver.size(); ++i) {
        solverData.add(UniqueArray<Real>());
        UniqueArray<Real>& currentSolverData = solverData[solverData.size() - 1];
        Real memory_before_solve = getAllocatedMemory();
        resetMaxMemory();

        info() << "****** SOLVER #" << i
               << " : "; // << linearSolverNodeList[i].attrValue("name");
        Alien::ILinearSolver* solver = options()->linearSolver[i];

        if (not solver->hasParallelSupport() and m_parallel_mng->commSize() > 1) {
          info() << "Current solver has not a parallel support for solving linear system "
                    ": skip it";
          currentSolverData.add(std::numeric_limits<Real>::quiet_NaN());
          currentSolverData.add(std::numeric_limits<Real>::quiet_NaN());
          currentSolverData.add(std::numeric_limits<Real>::quiet_NaN());
          currentSolverData.add(false);
          currentSolverData.add(0);
          currentSolverData.add(std::numeric_limits<Real>::quiet_NaN());
        } else {
          {
            Timer::Sentry ts(&timer);
            // MatrixExp mA(matrixA);
            // VectorExp vB(vectorB);
            solve(solver, matrixA, vectorB, vectorX, options()->builder[ibuilder]);
          }
          Alien::SolverStatus status = solver->getStatus();
          if (status.succeeded
              && options()->builder[ibuilder] != AlienTestOptionTypes::DirectBuilder) {
            SimpleCSRLinearAlgebra alg;
            Alien::Move::VectorData vectorR(space, m_vdist);
            vectorR.setBlockInfos(vectorX.block());
            alg.mult(matrixA, vectorX, vectorR);
            alg.axpy(-1., vectorB, vectorR);
            Real res = alg.norm2(vectorR);
            info() << "RES : " << res;
          }
          currentSolverData.add(timer.lastActivationTime());
          currentSolverData.add(getAllocatedMemory() - memory_before_solve);
          currentSolverData.add(getMaxMemory() - memory_before_solve);
          currentSolverData.add(status.succeeded);
          currentSolverData.add(status.iteration_count);
          currentSolverData.add(status.residual);
          solver->end();
        };
      }

    if (subDomain()->parallelMng()->commRank() == 0) {
      info() << "========= Alien Statistics Summary with "
             << builderNodeList[ibuilder].value() << " =============";
      printf("\tIndexManager time=%5.2f mem=%7.2f max_mem=%7.2f\n",
          index_manager_stat.time, index_manager_stat.memory,
          index_manager_stat.memory_max);
      printf("\tProfiler     time=%5.2f mem=%7.2f max_mem=%7.2f count=%d\n",
          profile_stat.time, profile_stat.memory, profile_stat.memory_max,
          profile_stat.allocation_count);
      for (Integer i = 0; i < inserter_stats.size(); ++i)
        printf("\tFilling %2d   time=%5.2f mem=%7.2f max_mem=%7.2f count=%d\n", i,
            inserter_stats[i].time, inserter_stats[i].memory,
            inserter_stats[i].memory_max, inserter_stats[i].allocation_count);

      if (!options()->buildingOnly() && m_vect_size > 0) {
        printf("\tSolver %2s : %6s %7s %7s %7s %6s %8s\n", "id", "time", "dmem", "mmem",
            "success", "#iter", "residual");

        Integer iglobal = 0;
        // printf("Builder
        // %s\n",options()->builder.enumValues()->nameOfValue(options()->builder[ibuilder],"").localstr());

        for (Integer isolver = 0; isolver < options()->linearSolver.size();
             ++isolver, ++iglobal) {
          const UniqueArray<Real>& c = solverData[iglobal];

          printf("\tSolver %2d : %6.2f %7.2f %7.2f %7.0f %6.0f %8.3g", isolver, c[0],
              c[1], c[2], c[3], c[4], c[5]);
          printf("\n");
        }
      }
    }
  }

  delete m_memory_tracker, m_memory_tracker = NULL;

  subDomain()->timeLoopMng()->stopComputeLoop(true);
}

/*---------------------------------------------------------------------------*/

void
AlienTestModule::buildAndFillInBlockVector(Alien::Move::VectorData& vectorB,
    ConstArrayView<Integer> allXIndex, ConstArrayView<Real> value)
{
  Alien::Move::LocalBlockVectorWriter b(std::move(vectorB));
  ENUMERATE_CELL (icell, m_areaT.own()) {
    const Integer id = allXIndex[icell->localId()];
    b[id][0] = value[0];
  }

  ENUMERATE_CELL (icell, m_areaP.own()) {
    const Integer id = allXIndex[icell->localId()];
    for (Integer i = 1; i < value.size(); ++i)
      b[id][i] = value[i];
  }
  vectorB = b.release();
}

void
AlienTestModule::buildAndFillInBlockVector(Alien::Move::VectorData& vectorB,
    ConstArrayView<Integer> allUIndex, ConstArray2View<Integer> allPIndex,
    ConstArrayView<Integer> allXIndex ALIEN_UNUSED_PARAM, ConstArrayView<Real> value)
{
  Alien::Move::LocalBlockVectorWriter b(std::move(vectorB));
  ENUMERATE_CELL (icell, m_areaT.own()) {
    const Integer id = allUIndex[icell->localId()];
    b[id][0] = value[0];
  }

  ENUMERATE_CELL (icell, m_areaP.own()) {
    const Integer id = allUIndex[icell->localId()];
    for (Integer i = 1; i < value.size(); ++i)
      b[id][i] = value[i];
  }

  ENUMERATE_NODE (inode, ownNodes()) {
    const Integer id0 = allPIndex[inode->localId()][0];
    b[id0][0] = value[0];
    const Integer id1 = allPIndex[inode->localId()][1];
    b[id1][0] = value[0];
  }
  vectorB = b.release();
}

void
AlienTestModule::buildAndFillInVector(Alien::Move::VectorData& vectorB, const double& value)
{
  const Alien::VectorDistribution& dist = vectorB.distribution();
  Alien::Move::LocalVectorWriter v(std::move(vectorB));
  info() << "Vector local size= " << v.size();
  for (Integer i = 0; i < dist.localSize(); ++i)
    v[i] = value;
  vectorB = v.release();
}

void
AlienTestModule::buildAndFillInBlockVector(
    Alien::Move::VectorData& vectorB, const double& value)
{
  const Alien::VectorDistribution& dist = vectorB.distribution();
  const Alien::Block* block = vectorB.block();
  Alien::Move::LocalBlockVectorWriter v(std::move(vectorB));
  for (Integer i = 0; i < dist.localSize(); ++i) {
    ArrayView<Real> values = v[i];
    for (Integer j = 0; j < block->size(); ++j)
      values[j] = value;
  }
  vectorB = v.release();
}

/*---------------------------------------------------------------------------*/

template <typename Profiler>
void
AlienTestModule::profileMatrix(const CellCellGroup& cell_cell_connection,
    const ItemGroup areaP, const UniqueArray2<Integer>& allPIndex,
    const Arccore::UniqueArray<Integer>& allUIndex,
    const Arccore::UniqueArray<Integer>& allXIndex, Profiler& profiler)
{
  ///////////////////////////////////////////////////////////////////////////
  //
  // DEFINE PROFILE
  //
  ENUMERATE_ITEMPAIR(Cell, Cell, icell, cell_cell_connection)
  {
    const Cell& cell = *icell;
    const Integer iIndex = allUIndex[cell.localId()];
    profiler.addMatrixEntry(iIndex, allUIndex[cell.localId()]);
    ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
    {
      const Cell& subcell = *isubcell;
      profiler.addMatrixEntry(iIndex, allUIndex[subcell.localId()]);
    }
  }

  ENUMERATE_NODE (inode, areaP.own()) {
    const Node& node = *inode;
    ConstArrayView<Integer> indexes = allPIndex[node.localId()];
    for (Integer i = 0; i < indexes.size(); ++i) {
      profiler.addMatrixEntry(indexes[i], indexes[i]);
    }
  }

  for (Integer localId = 0; localId < m_n_extra_indices; ++localId) {
    profiler.addMatrixEntry(allXIndex[localId], allXIndex[localId]);
  }
}

/*---------------------------------------------------------------------------*/

// template<typename StreamBuilderT>
// void
// AlienTestModule::
// streamProfileMatrix(const CellCellGroup& cell_cell_connection,
//    const ItemGroup areaU,
//    const ItemGroup areaP,
//    const UniqueArray2<Integer>& allPIndex,
//    const Alien::IntegerUniqueArray& allUIndex,
//    const Alien::IntegerUniqueArray& allXIndex,
//    StreamBuilderT & inserters)
//{
//  typename StreamBuilderT::Profiler & uInserter = inserters.getNewInserter(); // id 0
//  ARCANE_ASSERT((uInserter.getId() == 0),("Bad ID"));
//  typename StreamBuilderT::Profiler & pInserter = inserters.getNewInserter(); // id 1
//  ARCANE_ASSERT((pInserter.getId() == 1),("Bad ID"));
//  typename StreamBuilderT::Profiler & xInserter = inserters.getNewInserter(); // id 2
//  ARCANE_ASSERT((xInserter.getId() == 2),("Bad ID"));
//
//  ENUMERATE_ITEMPAIR(Cell,Cell,icell,cell_cell_connection)
//  {
//    const Cell & cell = *icell;
//    const Integer iIndex = allUIndex[cell.localId()];
//    uInserter.addMatrixEntry(iIndex, allUIndex[cell.localId()]);
//    ENUMERATE_SUB_ITEM(Cell,isubcell,icell)
//    {
//      const Cell& subcell = *isubcell;
//      uInserter.addMatrixEntry(iIndex,allUIndex[subcell.localId()]);
//    }
//  }
//
//  ENUMERATE_NODE(inode, areaP.own())
//  {
//    const Node & node = *inode;
//    ConstArrayView<Integer> indexes = allPIndex[node.localId()];
//    for(Integer i=0;i<indexes.size();++i)
//    {
//      pInserter.addMatrixEntry(indexes[i], indexes[i]);
//    }
//  }
//
//  for(Integer localId=0;localId<m_n_extra_indices;++localId)
//  {
//    xInserter.addMatrixEntry(allXIndex[localId],allXIndex[localId]);
//  }
//
//  inserters.allocate(); // optional if used with out-of-scope dtor
//}

/*---------------------------------------------------------------------------*/

// void
// AlienTestModule::
// streamBlockFillInMatrix(const CellCellGroup& cell_cell_connection,
//    const ItemGroup areaP,
//    const UniqueArray2<Integer>& allPIndex,
//    const Alien::IntegerUniqueArray& allUIndex,
//    const Alien::IntegerUniqueArray& allXIndex,
//    Alien::StreamMatrixBuilder & inserters)
//{
//  Alien::StreamMatrixBuilder::Filler & uInserter = inserters.getInserter(0);
//  uInserter.start();
//  Alien::StreamMatrixBuilder::Filler & pInserter = inserters.getInserter(1);
//  pInserter.start();
//  Alien::StreamMatrixBuilder::Filler & xInserter = inserters.getInserter(2);
//  xInserter.start();
//  Integer neq = m_vect_size;
//  Integer nuk = neq;
//
//  UniqueArray<Real> values(neq*nuk);
//  ENUMERATE_ITEMPAIR(Cell, Cell, icell, cell_cell_connection)
//  {
//    const Cell & cell = *icell;
//    values.fill(0.);
//    for (Integer i = 0; i < neq; ++i)
//      values[i*nuk+i] = fij(cell, cell);
//
//    uInserter.addBlockData(values);
//    ++uInserter;
//    ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
//    {
//      const Arcane::Cell& subcell = *isubcell;
//      //values.fill(fij(cell,subcell));
//      for (Integer k = 0; k < neq; ++k)
//        values[k*nuk+k] = fij(cell, subcell);
//      uInserter.addBlockData(values);
//      ++uInserter;
//    }
//  }
//
//  for (Integer i = 0; i < neq; ++i)
//    values[i*nuk+i] = 2;
//  ENUMERATE_NODE(inode, areaP.own())
//  {
//    const Node & node = *inode;
//    ConstArrayView<Integer> indexes = allPIndex[node.localId()];
//    for(Integer i=0;i<indexes.size();++i)
//    {
//      pInserter.addBlockData(values);
//      ++pInserter;
//    }
//  }
//
//  for (Integer i = 0; i < neq; ++i)
//    values[i*nuk+i] = 5;
//  for(Integer localId=0;localId<m_n_extra_indices;++localId)
//  {
//    xInserter.addBlockData(values);
//    ++xInserter;
//  }
//}
//
// void
// AlienTestModule::
// streamBlockFillInMatrix(const CellCellGroup& cell_cell_connection,
//    const ItemGroup areaP,
//    const UniqueArray2<Integer>& allPIndex,
//    const Alien::IntegerUniqueArray& allUIndex,
//    const Alien::IntegerUniqueArray& allXIndex,
//    Alien::StreamVBlockMatrixBuilder & inserters)
//{
//  Alien::StreamVBlockMatrixBuilder::Filler & uInserter = inserters.getInserter(0);
//  uInserter.start();
//  Alien::StreamVBlockMatrixBuilder::Filler & pInserter = inserters.getInserter(1);
//  pInserter.start();
//  Alien::StreamVBlockMatrixBuilder::Filler & xInserter = inserters.getInserter(2);
//  xInserter.start();
//  Integer neq = m_vect_size;
//  Integer nuk = neq;
//
//  UniqueArray2<Real> values ;
//  values.resize(neq,nuk);
//  ENUMERATE_ITEMPAIR(Cell, Cell, icell, cell_cell_connection)
//  {
//    const Cell & cell = *icell;
//    values.fill(0.);
//    for (Integer i = 0; i < neq; ++i)
//      values[i][i] = fij(cell, cell);
//
//    uInserter.addBlockData(values);
//    ++uInserter;
//    ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
//    {
//      const Arcane::Cell& subcell = *isubcell;
//      //values.fill(fij(cell,subcell));
//      for (Integer k = 0; k < neq; ++k)
//        values[k][k] = fij(cell, subcell);
//      uInserter.addBlockData(values);
//      ++uInserter;
//    }
//  }
//
//  for (Integer i = 0; i < neq; ++i)
//    values[i][i] = 2;
//  ENUMERATE_NODE(inode, areaP.own())
//  {
//    const Node & node = *inode;
//    ConstArrayView<Integer> indexes = allPIndex[node.localId()];
//    for(Integer i=0;i<indexes.size();++i)
//    {
//      pInserter.addBlockData(values);
//      ++pInserter;
//    }
//  }
//
//  for (Integer i = 0; i < neq; ++i)
//    values[i][i] = 5;
//  for(Integer localId=0;localId<m_n_extra_indices;++localId)
//  {
//    xInserter.addBlockData(values);
//    ++xInserter;
//  }
//}
//
// void
// AlienTestModule::
// streamFillInMatrix(const CellCellGroup& cell_cell_connection,
//    const ItemGroup areaP,
//    const UniqueArray2<Integer>& allPIndex,
//    const Alien::IntegerUniqueArray& allUIndex,
//    const Alien::IntegerUniqueArray& allXIndex,
//    Alien::StreamMatrixBuilder & inserters)
//{
//
//  Alien::StreamMatrixBuilder::Filler & uInserter = inserters.getInserter(0);
//  uInserter.start();
//  Alien::StreamMatrixBuilder::Filler & pInserter = inserters.getInserter(1);
//  pInserter.start();
//  Alien::StreamMatrixBuilder::Filler & xInserter = inserters.getInserter(2);
//  xInserter.start();
//  ENUMERATE_ITEMPAIR(Cell,Cell,icell,cell_cell_connection)
//  {
//    const Cell & cell = *icell;
//    uInserter.addData(fij(cell,cell));
//    ++uInserter;
//    ENUMERATE_SUB_ITEM(Cell,isubcell,icell)
//    {
//      const Cell& subcell = *isubcell;
//      uInserter.addData(fij(cell,subcell)) ;
//      ++uInserter;
//    }
//  }
//
//  ENUMERATE_NODE(inode, areaP.own())
//  {
//    const Node & node = *inode;
//    ConstArrayView<Integer> indexes = allPIndex[node.localId()];
//    for(Integer i=0;i<indexes.size();++i)
//    {
//      pInserter.addData(2);
//      ++pInserter;
//    }
//  }
//
//  for(Integer localId=0;localId<m_n_extra_indices;++localId)
//  {
//    xInserter.addData(5);
//    ++xInserter;
//  }
//}

/*---------------------------------------------------------------------------*/

template <typename Builder>
void
AlienTestModule::profiledFillInMatrix(const CellCellGroup& cell_cell_connection,
    const ItemGroup areaP, const UniqueArray2<Integer>& allPIndex,
    const Arccore::UniqueArray<Integer>& allUIndex,
    const Arccore::UniqueArray<Integer>& allXIndex, Builder& builder)
{
  ///////////////////////////////////////////////////////////////////////////
  //
  // FILL MATRIX
  //
  Integer neq = m_vect_size;
  Integer nuk = neq;
  UniqueArray2<Real> values(neq, nuk);
  ENUMERATE_ITEMPAIR(Cell, Cell, icell, cell_cell_connection)
  {
    const Cell& cell = *icell;
    values.fill(0.);
    for (Arcane::Integer i = 0; i < neq; ++i)
      values[i][i] = fij(cell, cell);

    Integer i = allUIndex[cell.localId()];
    builder(i, i) += values;
    ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
    {
      const Cell& subcell = *isubcell;
      for (Arcane::Integer k = 0; k < neq; ++k)
        values[k][k] = fij(cell, subcell);
      Integer j = allUIndex[subcell.localId()];
      builder(i, j) += values;
    }
  }

  values.fill(0.);
  ENUMERATE_NODE (inode, areaP.own()) {
    const Node& node = *inode;
    for (Arcane::Integer i = 0; i < neq; ++i)
      values[i][i] = 2;
    ConstArrayView<Integer> indexes = allPIndex[node.localId()];
    for (Integer i = 0; i < indexes.size(); ++i) {
      builder(indexes[i], indexes[i]) += values;
    }
  }

  for (Integer localId = 0; localId < m_n_extra_indices; ++localId) {
    for (Arcane::Integer i = 0; i < neq; ++i)
      values[i][i] = 5;
    builder(allXIndex[localId], allXIndex[localId]) += values;
  }
}

template <typename Builder>
void
AlienTestModule::fillInMatrix(const CellCellGroup& cell_cell_connection,
    const ItemGroup areaP, const UniqueArray2<Integer>& allPIndex,
    const Arccore::UniqueArray<Integer>& allUIndex,
    const Arccore::UniqueArray<Integer>& allXIndex, Builder& builder)
{
  ENUMERATE_ITEMPAIR(Cell, Cell, icell, cell_cell_connection)
  {
    const Cell& cell = *icell;
    const Integer iIndex = allUIndex[cell.localId()];
    builder(iIndex, iIndex) += fij(cell, cell);
    ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
    {
      const Cell& subcell = *isubcell;
      builder(iIndex, allUIndex[subcell.localId()]) += fij(cell, subcell);
    }
  }
  ENUMERATE_NODE (inode, areaP.own()) {
    const Node& node = *inode;
    ConstArrayView<Integer> indexes = allPIndex[node.localId()];
    for (Integer i = 0; i < indexes.size(); ++i) {
      builder(indexes[i], indexes[i]) += 2;
    }
  }

  for (Integer localId = 0; localId < m_n_extra_indices; ++localId) {
    builder(allXIndex[localId], allXIndex[localId]) += 5;
  }
}

/*---------------------------------------------------------------------------*/

Alien::SolverStatus
AlienTestModule::solve(Alien::ILinearSolver* solver, Alien::Move::MatrixData& matrixA,
    Alien::Move::VectorData& vectorB, Alien::Move::VectorData& vectorX,
    AlienTestOptionTypes::eBuilderType builderType)
{
  { // Réinitialisation de vectorX
    Alien::Move::LocalVectorWriter v(std::move(vectorX));
    for (Integer i = 0; i < v.size(); ++i)
      v[i] = 0;
    vectorX = v.release();
  }

  solver->solve(matrixA, vectorB, vectorX);

  Alien::SolverStatus status = solver->getStatus();
  if (status.succeeded) {
    info() << "Solver succeeds with " << status.iteration_count << " iterations";

    ///////////////////////////////////////////////////////////////////////////
    //
    // OPERATIONS ALGEBRIQUES
    //
    const auto& row_space = matrixA.rowSpace();
    const auto& col_space = matrixA.colSpace();
    const Block* block = vectorB.block();
    const VBlock* vblock = vectorB.vblock();
    std::shared_ptr<Alien::ILinearAlgebra> alg = solver->algebra();
    if (alg) {
      Alien::Move::VectorData vectorY(col_space, m_vdist);
      Alien::Move::VectorData vectorRes(row_space, m_vdist);

      vectorY.setBlockInfos(block);
      vectorRes.setBlockInfos(block);
      vectorY.setBlockInfos(vblock);
      vectorRes.setBlockInfos(vblock);

      alg->mult(matrixA, vectorX, vectorY);

      // res = b - A.x
      alg->copy(vectorB, vectorRes);
      alg->axpy(-1., vectorY, vectorRes);

      Real norm = alg->norm2(vectorRes);
      info() << "Residual norm2 by solver linear algebra : " << norm;
    }

    {
      if (builderType != AlienTestOptionTypes::DirectBuilder) {
        Alien::SimpleCSRLinearAlgebra testAlg;
        Alien::Move::VectorData vectorY(col_space, m_vdist);
        Alien::Move::VectorData vectorRes(row_space, m_vdist);

        vectorY.setBlockInfos(block);
        vectorRes.setBlockInfos(block);
        vectorY.setBlockInfos(vblock);
        vectorRes.setBlockInfos(vblock);
        testAlg.mult(matrixA, vectorX, vectorY);
        // res = b - A.x
        testAlg.copy(vectorB, vectorRes);
        testAlg.axpy(-1., vectorY, vectorRes);
        Real norm = testAlg.norm2(vectorRes);
        info() << "Residual norm2 by SimpleCSR algebra : " << norm;
      }
    }

#ifdef ALIEN_USE_PETSC
    {
      Alien::PETScLinearAlgebra testAlg;
      Alien::Move::VectorData vectorY(col_space, m_vdist);
      Alien::Move::VectorData vectorRes(row_space, m_vdist);

      vectorY.setBlockInfos(block);
      vectorRes.setBlockInfos(block);
      vectorY.setBlockInfos(vblock);
      vectorRes.setBlockInfos(vblock);
      testAlg.mult(matrixA, vectorX, vectorY);
      // res = b - A.x
      testAlg.copy(vectorB, vectorRes);
      testAlg.axpy(-1., vectorY, vectorRes);
      Real norm = testAlg.norm2(vectorRes);
      info() << "Residual norm2 by PETSc algebra : " << norm;
    }
#endif // ALIEN_USE_PETSC
  } else {
    fatal() << "Solver fails after " << status.iteration_count << " iterations";
  }

  return status;
}

/*---------------------------------------------------------------------------*/

void
AlienTestModule::checkVectorValues(Alien::IVector& vectorX, const double& value)
{
  // const VectorDistribution& dist = vectorX.distribution();
  const Block* block = vectorX.impl()->block();
  const VBlock* vblock = vectorX.impl()->vblock();
  auto& vx = dynamic_cast<Alien::Move::VectorData&>(vectorX);
  if (block) {
    const Alien::Move::LocalBlockVectorReader v(vx);
    for (Integer i = 0; i < m_vdist.localSize(); ++i) {
      ConstArrayView<Real> values = v[i];
      for (Integer j = 0; j < block->size(); ++j) {
        if (values[j] != value)
          fatal() << "Incorrect value (" << values[j] << " vs expected 2.)";
      }
    }
  } else if (vblock)
    throw FatalErrorException(A_FUNCINFO, "Not implemented yet");
  else {
    const Alien::Move::LocalVectorReader v(vx);
    for (Integer i = 0; i < m_vdist.localSize(); ++i)
      if (v[i] != value)
        fatal() << "Incorrect value (" << v[i] << " vs expected 2.)";
  }
  // auto-call for vB.end() in const context
}

/*---------------------------------------------------------------------------*/

void
AlienTestModule::vectorVariableUpdate(Alien::Move::VectorData& vectorB,
    Alien::ArcaneTools::IIndexManager::ScalarIndexSet indexSetU)
{
  ///////////////////////////////////////////////////////////////////////////
  //
  // VECTOR / VARIABLE
  //
  // jmg Alien::ItemVectorAccessor v(std::move(vectorB));
  {
    Alien::ArcaneTools::ItemVectorAccessor v(vectorB);
    if (m_vect_size == 1) {
      // remplissage algébrique de vectorB : accessor / view
      Alien::ArcaneTools::Variable(m_u) = v(indexSetU);
      Alien::ArcaneTools::Variable(m_w)[1] = v(indexSetU);
      Alien::ArcaneTools::Variable(m_u) += v(indexSetU);
      v(indexSetU) = 0;
      v(indexSetU) = Alien::ArcaneTools::Variable(m_u);
      v(indexSetU) -= Alien::ArcaneTools::Variable(m_w)[1];
    } else if (m_vect_size > 1) {
      Alien::ArcaneTools::Variable(m_w) = v(indexSetU);
      v(indexSetU) = 0.;
      v(indexSetU) = Alien::ArcaneTools::Variable(m_w);
    }
    // jmg vectorB = v.release();
  }
}

/*---------------------------------------------------------------------------*/

void
AlienTestModule::checkDotProductWithManyAlgebra(Alien::IVector& vectorB,
    Alien::IVector& vectorX, Alien::Space& space ALIEN_UNUSED_PARAM)
{
  info() << "Checking dot product using two linear algebra implementations";

  Alien::SimpleCSRLinearAlgebra alg1;
  alg1.copy(vectorB, vectorX); // copy b in x
  const Real dot1 = alg1.dot(vectorB, vectorX);
  info() << "SimpleCSR dot product : " << dot1;
#ifdef ALIEN_USE_PETSC
  Alien::PETScLinearAlgebra alg2;
  alg2.copy(vectorB, vectorX); // copy b in x
  const Real dot2 = alg2.dot(vectorB, vectorX);
  info() << "PETSc dot product : " << dot2;
#ifdef ALIEN_USE_MTL4
  if (m_parallel_mng->commSize() == 1) {
    Alien::MTLLinearAlgebra alg3;
    alg3.copy(vectorB, vectorX); // copy b in x
    const Real dot3 = alg3.dot(vectorB, vectorX);
    info() << "MTL dot product : " << dot3;

    if (dot1 != dot2 || dot1 != dot3)
      fatal() << "Not equal dot product";
  } else
#endif // ALIEN_USE_MTL4
  {
    if (dot1 != dot2)
      fatal() << "Not equal dot product";
  }
#endif // ALIEN_USE_PETSC
}

/*---------------------------------------------------------------------------*/

Real
AlienTestModule::fij(const Cell& ci, const Cell& cj)
{
  if (ci == cj)
    return m_diag_coeff;
  else
    return -1;
}

/*---------------------------------------------------------------------------*/

Real
AlienTestModule::getAllocatedMemory()
{
  //   // Utilise IMemoryInfo, si ca marche ....
  //   std::ostringstream oss;
  //   subDomain()->memoryInfo()->printAllocatedMemory(oss,globalIteration());
  //   long int current_allocated;
  //   if (sscanf(oss.str().c_str()," INFO_ALLOCATION: current= %ld ",&current_allocated)
  //   != 1)
  //     return -1;
  //   return Real(current_allocated)/1048576;

  // Version a la mano
  if (options()->checkMemory()) {
    ARCANE_ASSERT((m_memory_tracker != NULL), ("Invalid null memory tracker"));
    return Real(m_memory_tracker->getTotalAllocation()) / 1048576;
  } else {
    // Version très approximative mais ne coutant rien
    return platform::getMemoryUsed() / 1048576;
  }
}

/*---------------------------------------------------------------------------*/

Real
AlienTestModule::getMaxMemory()
{
  // Version a la mano
  if (options()->checkMemory()) {
    ARCANE_ASSERT((m_memory_tracker != NULL), ("Invalid null memory tracker"));
    return Real(m_memory_tracker->getPeakAllocation()) / 1048576;
  } else {
    // Version très approximative mais ne coutant rien
    return std::numeric_limits<Real>::quiet_NaN();
  }
}

/*---------------------------------------------------------------------------*/

void
AlienTestModule::resetMaxMemory()
{
  if (m_memory_tracker)
    m_memory_tracker->resetPeakAllocation();
}

/*---------------------------------------------------------------------------*/

Int64
AlienTestModule::getAllocationCount()
{
  if (m_memory_tracker)
    return m_memory_tracker->getAllocationCount()
        + m_memory_tracker->getReallocationCount();
  else
    return -1;
}

ARCANE_REGISTER_MODULE_ALIENTEST(AlienTestModule);
