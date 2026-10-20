using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Scriban;
using Scriban.Runtime;

namespace GeometricGenerationScriban
{
  /// <summary>
  /// Generation of the 'Generated*.h' files of arcane/src/arcane/geometry
  /// from the Scriban templates of the 'Templates' directory.
  /// This generator produces the same files as the C++26 reflection
  /// generator 'GeometricGenerationCpp'. The generated headers are plain
  /// C++ and can be used by any C++20 compiler.
  /// </summary>
  public static class MainClass
  {
    public static int Main(string[] args)
    {
      GeomShapeInfos.Check();

      string base_path = (args.Length > 0) ? args[0] : "../../src/arcane/geometry";
      string template_dir = Path.Combine(AppContext.BaseDirectory, "Templates");

      _Generate(template_dir, "GeneratedConnectivity.h.scriban", base_path, "GeneratedConnectivity.h");
      _Generate(template_dir, "GeneratedGeomShapeView.h.scriban", base_path, "GeneratedGeomShapeView.h");
      _Generate(template_dir, "GeneratedGeomElement.h.scriban", base_path, "GeneratedGeomElement.h");
      _Generate(template_dir, "GeneratedGeomElementView.h.scriban", base_path, "GeneratedGeomElementView.h");
      _Generate(template_dir, "GeneratedGeomShapeViewDeclarations.h.scriban", base_path, "GeneratedGeomShapeViewDeclarations.h");
      return 0;
    }

    static void _Generate(string template_dir, string template_name, string base_path, string output_name)
    {
      string template_path = Path.Combine(template_dir, template_name);
      var template = Template.Parse(File.ReadAllText(template_path, Encoding.UTF8), template_path);
      if (template.HasErrors) {
        foreach (var error in template.Messages)
          Console.Error.WriteLine("Template error: {0}", error);
        throw new InvalidOperationException("Failed to parse template '" + template_path + "'");
      }

      var context = new TemplateContext();
      context.PushGlobal(_BuildModel());
      string text = template.Render(context);

      string full_path = Path.Combine(base_path, output_name);
      Console.WriteLine("Generating file '{0}'", full_path);
      File.WriteAllText(full_path, text, new UTF8Encoding(true));
    }

    /// <summary>
    /// Builds the model exposed to the templates: the objects are ScriptObject
    /// with explicit keys (name, nb_node, ...) so that the rendering does not
    /// depend on the .NET member renaming rules.
    /// </summary>
    static ScriptObject _BuildModel()
    {
      var root = new ScriptObject();
      root["items"] = _ToList(GeomShapeInfos.Items);
      root["items_2d_3d"] = _ToList(GeomShapeInfos.Items2DAnd3D);
      return root;
    }

    static List<ScriptObject> _ToList(IReadOnlyList<GeomShapeInfo> descriptors)
    {
      var result = new List<ScriptObject>();
      foreach (var info in descriptors) {
        var obj = new ScriptObject();
        obj["name"] = info.Name;
        obj["nb_node"] = info.NbNode;
        obj["nb_edge"] = info.NbEdge;
        obj["nb_face"] = info.NbFace;
        obj["dimension"] = info.Dimension;
        obj["basic_name"] = info.BasicName;
        obj["svc_name"] = info.SvcName;
        obj["coords_args"] = info.CoordsArgs;
        result.Add(obj);
      }
      return result;
    }
  }
}
