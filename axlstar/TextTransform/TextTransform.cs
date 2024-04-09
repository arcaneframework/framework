// 
// Main.cs
//  
// Author:
//       Mikayla Hutchinson <m.j.hutchinson@gmail.com>
// 
// Copyright (c) 2009 Novell, Inc. (http://www.novell.com)
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

using System;
using System.IO;

namespace Mono.TextTemplating
{
  class TextTransform
  {
    public static int Main (string[] args)
    {
      try {
        return MainInternal(args);
      }
      catch (Exception e) {
        Console.Error.WriteLine(e);
        return -1;
      }
    }

    static int MainInternal(string[] args)
    {
      string input_file = "";
      string class_namespace = "";
      if (args.Length == 1){
        class_namespace = "Arcane.Axl";
        input_file = args[0];
      }
      else if (args.Length == 3 && args[0] == "--namespace"){
        class_namespace = args[1];
        input_file = args[2];
      }
      else{
        Console.WriteLine("Usage: program --namespace name file.tt");
        return 1;
      }

      var generator = new TemplateGenerator();
      if (!File.Exists(input_file)) {
        Console.Error.WriteLine("Input file '{0}' does not exist.", input_file);
        return -1;
      }
      string class_name = Path.GetFileNameWithoutExtension(input_file);
      string input_dir = Path.GetDirectoryName(input_file);
      string output_file = Path.Combine(input_dir, class_name + ".cs");

      Console.WriteLine("Generating class name '{0}' with namespace '{1}'", class_name, class_namespace);
      Console.WriteLine("Generating file '{0}'", output_file);

      var out_encoding = new System.Text.UTF8Encoding();
      generator.PreprocessTemplate(input_file, class_name, class_namespace, output_file, out_encoding,
        out string language, out string[] references);
      if (generator.Errors.HasErrors) {
        Console.Write("Preprocessing '{0}' into class '{1}.{2}' failed.", input_file, class_namespace, class_name);
      }


      foreach (System.CodeDom.Compiler.CompilerError err in generator.Errors)
        Console.Error.WriteLine("{0}({1},{2}): {3} {4}", err.FileName, err.Line, err.Column,
                           err.IsWarning ? "WARNING" : "ERROR", err.ErrorText);

      return generator.Errors.HasErrors ? -1 : 0;
    }
  }
}
