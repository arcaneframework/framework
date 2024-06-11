%module(directors="1") ArcaneServices

%import core/ArcaneSwigCore.i

%{
#include "ArcaneSwigUtils.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/IUnitTest.h"
#include "arcane/core/IDataReader.h"
#include "arcane/core/IDataWriter.h"
#include "arcane/core/ICaseFunctionProvider.h"
#include "arcane/core/ICheckpointWriter.h"
#include "arcane/core/ICheckpointReader.h"
#include "arcane/core/IVariableReader.h"
#include "arcane/core/IDirectExecution.h"
#include "arcane/core/ITimeHistoryCurveWriter2.h"
#include "arcane/core/IPostProcessorWriter.h"
#include "arcane/core/IExternalPlugin.h"
using namespace Arcane;
%}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_SWIG_DEFINE_SERVICE(Arcane,IDataWriter,
                           public abstract void BeginWrite(VariableCollection vars);
                           public abstract void EndWrite();
                           public abstract void SetMetaData(string meta_data);
                           public abstract void Write(IVariable var, IData data);
                           );

ARCANE_SWIG_DEFINE_SERVICE(Arcane,IDataReader,
                           public abstract void BeginRead(VariableCollection vars);
                           public abstract void EndRead();
                           public abstract string MetaData();
                           public abstract void Read(IVariable var, IData data);
                           );

ARCANE_SWIG_DEFINE_SERVICE(Arcane,IUnitTest,
                           public abstract void InitializeTest();
                           public abstract void ExecuteTest();
                           public abstract void FinalizeTest();
                           );

ARCANE_SWIG_DEFINE_SERVICE(Arcane,ICaseFunctionProvider,
                           public abstract void RegisterCaseFunctions(ICaseMng cm);
                           );

ARCANE_SWIG_DEFINE_SERVICE(Arcane,ICaseFunctionDotNetProvider,
                           public abstract void RegisterCaseFunctions(ICaseMng cm,
                                                                      string assembly_name,
                                                                      string class_name);
                           );

ARCANE_SWIG_DEFINE_SERVICE(Arcane,ICheckpointWriter,
                           public abstract IDataWriter DataWriter();
                           public abstract void NotifyBeginWrite();
                           public abstract void NotifyEndWrite();
                           public abstract void SetFileName(string file_name);
                           public abstract string FileName();
                           public abstract void SetBaseDirectoryName(string dirname);
                           public abstract string BaseDirectoryName();
                           public abstract void SetCheckpointTimes(Arcane.RealConstArrayView times);
                           public abstract Arcane.RealConstArrayView CheckpointTimes();
                           public abstract void Close();
                           public abstract string ReaderServiceName();
                           public abstract string ReaderMetaData();
                           );

ARCANE_SWIG_DEFINE_SERVICE(Arcane,ICheckpointReader,
                           public abstract IDataReader DataReader();
                           public abstract void NotifyBeginRead();
                           public abstract void NotifyEndRead();
                           public abstract void SetFileName(string file_name);
                           public abstract string FileName();
                           public abstract void SetBaseDirectoryName(string dirname);
                           public abstract string BaseDirectoryName();
                           public abstract void SetReaderMetaData(string arg0);
                           public abstract void SetCurrentTimeAndIndex(double current_time, int current_index);
                           );

ARCANE_SWIG_DEFINE_SERVICE(Arcane,ITimeHistoryCurveWriter2,
                           public abstract void Build();
                           public abstract void BeginWrite(TimeHistoryCurveWriterInfo infos);
                           public abstract void EndWrite();
                           public abstract void WriteCurve(TimeHistoryCurveInfo infos);
                           public abstract string Name();
                           public abstract void SetOutputPath(string path);
                           public abstract string OutputPath();
                           );

ARCANE_SWIG_DEFINE_SERVICE(Arcane,IPostProcessorWriter,
                           public abstract void Build();
                           public abstract IDataWriter DataWriter();
                           public abstract void SetBaseDirectoryName(string dirname);
                           public abstract string BaseDirectoryName();
                           public abstract void SetBaseFileName(string filename);
                           public abstract string BaseFileName();
                           public abstract void SetTimes(Arcane.RealConstArrayView times);
                           public abstract Arcane.RealConstArrayView Times();
                           public abstract void SetVariables(VariableCollection variables);
                           public abstract VariableCollection Variables();
                           public abstract void SetGroups(ItemGroupCollection groups);
                           public abstract ItemGroupCollection Groups();
                           public abstract void NotifyBeginWrite();
                           public abstract void NotifyEndWrite();
                           public abstract void Close();
                           );

ARCANE_SWIG_DEFINE_SERVICE(Arcane,IDirectExecution,
                           public abstract void Build();
                           public abstract void Execute();
                           public abstract bool IsActive();
                           public abstract void SetParallelMng(IParallelMng pm);
                           );

ARCANE_SWIG_DEFINE_SERVICE(Arcane,IVariableReader,
                           public abstract void SetBaseDirectoryName(string path);
                           public abstract void SetBaseFileName(string filename);
                           public abstract void Initialize(bool is_start);
                           public abstract void SetVariables(VariableCollection vars);
                           public abstract void UpdateVariables(double wanted_time);
                           public abstract Real2 TimeInterval(IVariable var);
                           );

ARCANE_SWIG_DEFINE_SERVICE(Arcane,IExternalPlugin,
                           public abstract void LoadFile(string filename);
                           public abstract void ExecuteFunction(string function_name);
                           );
