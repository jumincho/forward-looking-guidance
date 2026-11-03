from canonical_answer import format_answer

assert format_answer('Where was Mira born?','Eastbrook, North Province')=='Eastbrook'
assert format_answer('What is the place of death of Iven?','Stonehaven, West County, Aridia')=='Stonehaven'
assert format_answer('Where was Mira born?','Fictional City, D.C., Aridia')=='Fictional City, D.C.'
assert format_answer('Where was Mira born?','12 Garden Road, Eastbrook')=='12 Garden Road, Eastbrook'
assert format_answer('Who founded the company?','Sir Iven Vale')=='Iven Vale'
assert format_answer('Who founded the company?','Dame Mira Lane')=='Mira Lane'
assert format_answer('Who is the ruler?','Adalbert I, Margrave of Westland')=='Adalbert I, Margrave of Westland'
assert format_answer('Which book was published first?','The North, The South')=='The North, The South'
assert format_answer('Where is the district?','Eastbrook, North Province')=='Eastbrook, North Province'
assert format_answer('Who is the character?','Sir Lancelot')=='Sir Lancelot'
assert format_answer('Where was Mira born?',None) is None
print('CANONICAL_ANSWER_SYNTHETIC_CHECKS_PASSED')
