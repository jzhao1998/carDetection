CREATE database company;
CREATE TABLE tableregister(
    tablenum int,
    waiterid int,
    menuid int not null,
    inuse boolean,
    primary key(menuid)
);
CREATE TABLE plat(
    platid int not null,
    platname varchar(100) not null,
    descriptionword varchar(100),
    ingredient varchar(100),
    imageplatpath1 varchar(100),
    imageplatpath2 varchar(100),
    imageplatpath3 varchar(100),
    prohibited varchar(100),
    timecook int,
    primary key(platid)
);
CREATE TABLE menu(
    id int not null,
    menuid int not null,
    platid int not null,
    ready boolean,
    numbers int,
    primary key(id)
);
