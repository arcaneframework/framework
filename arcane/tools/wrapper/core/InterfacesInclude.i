// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-

%feature("director") Arcane::IExternalPlugin;

ARCANE_STD_EXHANDLER
%include arcane/core/IUnitTest.h
%include arcane/core/IDataWriter.h
%include arcane/core/IDataReader.h
%include arcane/core/IMeshReader.h
%include arcane/core/ICaseFunctionProvider.h
%include arcane/core/ICheckpointWriter.h
%include arcane/core/ICheckpointReader.h
%include arcane/core/IVariableReader.h
%include arcane/core/IDirectExecution.h
%include arcane/core/ITimeHistoryCurveWriter2.h
%include arcane/core/IPostProcessorWriter.h
%include arcane/core/IExternalPlugin.h
%exception;
