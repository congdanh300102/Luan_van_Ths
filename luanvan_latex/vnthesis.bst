%% vnthesis.bst — kiểu danh mục tài liệu tham khảo theo quy cách luận văn Việt Nam
%%
%%  * Tài liệu tiếng Việt (trường  language = {vietnamese}) xếp trước, tài liệu
%%    tiếng nước ngoài xếp sau; mỗi nhóm có tiêu đề riêng.
%%  * Tác giả người Việt (trong tài liệu tiếng Việt) giữ nguyên thứ tự họ tên
%%    "Nguyễn Tiến Hưng" và được xếp theo tên; thứ tự xếp lấy từ trường
%%    sortkey (BibTeX không sắp được chữ có dấu theo bảng chữ cái tiếng Việt).
%%  * Tác giả nước ngoài ghi "Họ, chữ cái đầu tên." và xếp theo họ.
%%  * Mỗi mục: Tác giả (năm), "Tên bài", Tên tạp chí, tập(số), trang.
%%    Sách, tên văn bản, trang web: tên in nghiêng.
%%  * Trích dẫn dạng số [n] theo thứ tự trong danh mục.

ENTRY
  { address author booktitle edition howpublished institution journal
    language note number pages publisher sortkey title volume year }
  {}
  { sort.label }

INTEGERS { nameptr namesleft numnames isvi lastgroup multiresult }
STRINGS { s t acc }

FUNCTION {not} { { #0 } { #1 } if$ }
FUNCTION {and} { 'skip$ { pop$ #0 } if$ }
FUNCTION {or}  { { pop$ #1 } 'skip$ if$ }

FUNCTION {set.lang}
{ language empty$
    { #0 'isvi := }
    { language "vietnamese" = { #1 'isvi := } { #0 'isvi := } if$ }
  if$
}

FUNCTION {emphasize}
{ duplicate$ empty$ { pop$ "" } { "\emph{" swap$ * "}" * } if$ }

FUNCTION {quoted}
{ duplicate$ empty$ { pop$ "" } { "``" swap$ * "''" * } if$ }

% Ghép các thành phần bằng dấu phẩy, bỏ qua thành phần rỗng
FUNCTION {add}
{ duplicate$ empty$
    { pop$ }
    { 's := acc empty$ { s 'acc := } { acc ", " * s * 'acc := } if$ }
  if$
}

FUNCTION {format.names}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr
      isvi { "{vv~}{ll}{ ff}{, jj}" } { "{vv~}{ll}{, f.}{, jj}" } if$
      format.name$ 't :=
      nameptr #1 >
        { namesleft #1 >
            { ", " * t * }
            { t "others" =
                { isvi { " và cộng sự" } { " et al." } if$ * }
                { isvi { " và " } { numnames #2 > { ", and " } { " and " } if$ } if$ * t * }
              if$
            }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {format.year}
{ year empty$
    { isvi { "(không ghi năm)" } { "(n.d.)" } if$ }
    { "(" year * ")" * }
  if$
}

% "Tác giả (năm)" — luôn là thành phần đầu tiên
FUNCTION {head}
{ author empty$
    { format.year }
    { author format.names " " * format.year * }
  if$
  'acc :=
}

FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
    { multiresult not t empty$ not and }
    { t #1 #1 substring$
      duplicate$ "-" =
      swap$ duplicate$ "," =
      swap$ "+" =
      or or
        { #1 'multiresult := }
        { t #2 global.max$ substring$ 't := }
      if$
    }
  while$
  multiresult
}

FUNCTION {format.pages}
{ pages empty$
    { "" }
    { isvi
        { "tr. " pages * }
        { pages multi.page.check { "pp. " pages * } { "p. " pages * } if$ }
      if$
    }
  if$
}

% Tạp chí: tiếng Việt "Tập x, Số y"; tiếng Anh "x(y)"
FUNCTION {format.volnum}
{ isvi
    { volume empty$ { "" } { "Tập " volume * } if$
      number empty$ 'skip$
        { duplicate$ empty$ { pop$ "Số " number * } { ", Số " * number * } if$ }
      if$
    }
    { volume empty$
        { number empty$ { "" } { "no. " number * } if$ }
        { volume number empty$ 'skip$ { "(" * number * ")" * } if$ }
      if$
    }
  if$
}

FUNCTION {format.edition}
{ edition empty$
    { "" }
    { isvi { "Tái bản lần thứ " edition * } { edition " edn." * } if$ }
  if$
}

FUNCTION {group.heading}
{ isvi lastgroup =
    'skip$
    { isvi
        { "\bibgroup{Tài liệu tiếng Việt}" }
        { "\bibgroup{Tài liệu tiếng Anh}" }
      if$
      write$ newline$
      isvi 'lastgroup :=
    }
  if$
}

FUNCTION {begin.entry}
{ set.lang
  group.heading
  "\bibitem{" cite$ * "}" * write$ newline$
}

FUNCTION {finish.entry}
{ acc "." * write$ newline$ }

FUNCTION {article}
{ begin.entry
  head
  title quoted add
  journal emphasize add
  format.volnum add
  format.pages add
  note add
  finish.entry
}

FUNCTION {book}
{ begin.entry
  head
  title emphasize add
  format.edition add
  publisher add
  address add
  note add
  finish.entry
}

FUNCTION {inproceedings}
{ begin.entry
  head
  title quoted add
  booktitle empty$
    'skip$
    { isvi { "trong " } { "in " } if$ booktitle emphasize * add }
  if$
  volume empty$ 'skip$ { isvi { "Tập " } { "vol. " } if$ volume * add } if$
  format.pages add
  publisher add
  note add
  finish.entry
}

FUNCTION {misc}
{ begin.entry
  head
  title emphasize add
  institution add
  howpublished add
  note add
  finish.entry
}

FUNCTION {default.type} { misc }

READ

FUNCTION {sortify} { purify$ "l" change.case$ }

FUNCTION {sort.format.names}
{ 's :=
  s #1 "{vv{ } }{ll{ }}{  ff{ }}{  jj{ }}" format.name$ sortify
}

FUNCTION {presort}
{ set.lang
  isvi { "0" } { "1" } if$
  "    " *
  sortkey empty$
    { author empty$ { title sortify } { author sort.format.names } if$ }
    { sortkey }
  if$
  *
  "    " *
  year empty$ { "0000" } { year } if$ *
  "    " *
  title empty$ { "" } { title sortify } if$ *
  #1 entry.max$ substring$
  'sort.label :=
  sort.label 'sort.key$ :=
}

ITERATE {presort}

SORT

FUNCTION {begin.bib}
{ preamble$ empty$ 'skip$ { preamble$ write$ newline$ } if$
  "\providecommand{\bibgroup}[1]{\item[]\hspace*{-\labelwidth}\hspace*{-\labelsep}\textbf{#1}}"
  write$ newline$
  "\begin{thebibliography}{99}" write$ newline$
  #-1 'lastgroup :=
}

EXECUTE {begin.bib}

ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$ "\end{thebibliography}" write$ newline$ }

EXECUTE {end.bib}
