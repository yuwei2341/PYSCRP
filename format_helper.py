#!/usr/bin/env python3

# Hide code and prompt in jupyter notebook
# Usage:
  # from format_helper import hide_code
  # from IPython.display import HTML
  # HTML(hide_code)

hide_code = '''
    <link rel="stylesheet"
    href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css"
    integrity="sha384-1q8mTJOASx8j1Au+a5WDVnPi2lkFfwwEAa8hDDdjZlpLegxhjVME1fgjWPGmkzs7"
    crossorigin="anonymous">
    <script>
      function code_toggle() {
        if (code_shown){
          $('div.input').hide('500');
          $('#toggleButton').val('Show Code')
        } else {
          $('div.input').show('500');
          $('#toggleButton').val('Hide Code')
        }
        code_shown = !code_shown
      }
      function prompt_toggle() {
        if (prompt_shown){
          $('div.prompt').hide('25');
          $('#togglePrompt').val('Show Prompt')
        } else {
          $('div.prompt').show('25');
          $('#togglePrompt').val('Hide Prompt')
        }
        prompt_shown = !prompt_shown
      }

      $( document ).ready(function(){
        code_shown=false;
        prompt_shown=true;
        $('div.input').hide()
        $('div.prompt').hide()
      });
    </script>
    <form action="javascript:code_toggle()">
        <input type="submit" class="btn btn-sm btn-default" id="toggleButton" value="Show Code" style="float: right;">
    </form>
    <form action="javascript:prompt_toggle()">
        <input type="submit" class="btn btn-sm btn-default" id="togglePrompt" value="Show Prompt" style="float: right;">
    </form>
'''

def print_table(table):
    # table is a list of tuples, where each tuple is a line, each element of the tuple is a column
    col_width = [max(len(x) for x in col) for col in zip(*table)]
    for line in table:
        print("| " + " ".join("{txt:{width}}".format(txt=x, width=col_width[i])
                                for i, x in enumerate(line)) + " |")
