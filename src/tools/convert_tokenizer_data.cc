#include "sslib/path.h"
#include "sslib/json.h"
#include "transformer/model_writer.h"

using namespace std;
using namespace sslib;
using namespace inferflow;
using namespace inferflow::transformer;

bool Convert(const string &input_path, const string &output_path)
{
    JsonParser jparser;
    jparser.Init();

    string content;
    bool ret = Path::GetFileContent_Text(content, input_path);
    Macro_RetxFalseIf(!ret, LogError("Failed to get the file content."));

    JsonDoc jdoc;
    ret = jparser.ParseUtf8(jdoc, content);
    Macro_RetxFalseIf(!ret, LogError("Invalid JSON format"));

    StdVocabulary vocab;
    string str;
    JsonObject jobj = jdoc.GetJObject();
    for (uint32_t field_idx = 0; field_idx < jobj.size; field_idx++)
    {
        const JsonObjectEntry &field = jobj.items[field_idx];
        field.name.ToString(str);

        StdVocabulary::Token token;
        token.id = atoi(str.c_str());
        field.value.GetString(token.str);

        vocab.token_array.push_back(token);
        vocab.str_to_id[token.str] = token.id;
    }

    ofstream writer(output_path);
    Macro_RetxFalseIf(!writer, LogError("Failed to open the output file"));

    ModelWriter::VocabularyToJson(writer, vocab);
    writer.close();

    return ret;
}

int main(int argc, const char *argv[])
{
    if (argc < 3) {
        cout << "convert_tokenizer_data <input_path> <output_path>" << endl;
        return 1;
    }

    string input_path = argv[1];
    string output_path = argv[2];

    bool ret = Convert(input_path, output_path);
    return ret ? 0 : 1;
}

