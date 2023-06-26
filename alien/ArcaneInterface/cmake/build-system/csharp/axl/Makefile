MCS=gmcs

SRCS_COMMON = AXLGlobal.cs AXLParser.cs CaseOptionsGenerator.cs CodeGenerator.cs ComplexOptionInfo.cs EntryPointInfo.cs EnumerationOptionInfo.cs EnumValueOptionInfo.cs ExtendedOptionInfo.cs GeneratorVisitor.cs ItemKind.cs ModuleBaseGenerator.cs ModuleInfo.cs NameTranslationsInfo.cs OptionInfo.cs OptionInfoVisitor.cs ScriptOptionInfo.cs ServiceBaseGenerator.cs ServiceInfo.cs ServiceInstanceOptionInfo.cs SimpleOptionInfo.cs Utils.cs VariableInfo.cs XmlInfo.cs ServiceOrModuleInfo.cs CSharpModuleGenerator.cs CSharpCodeGenerator.cs CppCodeGenerator.cs CSharpCaseOptionsGenerator.cs CSharpGeneratorVisitor.cs CSharpServiceGenerator.cs AxlEntities.cs IAXLParser.cs IAXLObjectFactory.cs DefaultAXLObjectFactory.cs

SRCS_AXL2CC = ${SRCS_COMMON} MainAXL2CC.cs 

SRCS_AXLDOC = ${SRCS_COMMON} CodeInfo.cs DoxygenDocumentationGenerator.cs DoxygenDocumentationFile.cs DoxygenDocumentationVisitor.cs DoxygenDocumentationUtils.cs DoxygenExampleVisitor.cs DoxygenExampleFile.cs DoxygenOptionIndexVisitor.cs MainAXLDOC.cs FinalAxlGenerator.cs

SRCS_AXLCOPY = ${SRCS_COMMON} MainAXLCOPY.cs 

DISTRIB_SRC_PATH = .

FULL_SRCS_AXL2CC = $(SRCS_AXL2CC:%=${DISTRIB_SRC_PATH}/Arcane.Axl/%)
FULL_SRCS_AXLDOC = $(SRCS_AXLDOC:%=${DISTRIB_SRC_PATH}/Arcane.Axl/%)
FULL_SRCS_AXLCOPY = $(SRCS_AXLCOPY:%=${DISTRIB_SRC_PATH}/Arcane.Axl/%)

all_new :
	xbuild Arcane.Axl.sln

all: bin/axl2cc.exe bin/axldoc.exe bin/axlcopy.exe

MCS_FLAGS := -debug+ -codepage:utf8 -resource:${DISTRIB_SRC_PATH}/axl.xsd,axl.xsd /r:System.Configuration

bin/axl2cc.exe : $(FULL_SRCS_AXL2CC) Makefile
	${MCS} $(FULL_SRCS_AXL2CC) -out:$@ ${MCS_FLAGS}

bin/axldoc.exe: $(FULL_SRCS_AXLDOC) Makefile
	${MCS} $(FULL_SRCS_AXLDOC) -out:$@ ${MCS_FLAGS}

bin/axlcopy.exe: $(FULL_SRCS_AXLCOPY) Makefile
	${MCS} $(FULL_SRCS_AXLCOPY) -out:$@ ${MCS_FLAGS}

clean:
	-rm bin/axl2cc.exe bin/axldoc.exe bin/axlcopy.exe